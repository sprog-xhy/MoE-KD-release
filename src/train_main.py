import argparse
import json
import math
import os.path
import pickle
import time
import sys
import deepspeed
import torch
import transformers
from deepspeed import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from termcolor import colored
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    GenerationConfig, AutoConfig
)

from dschat.utils.data.data_utils2 import create_sft_dataset
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
# from dschat.utils.data.data_utils import create_prompt_dataset
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, load_hf_tokenizer
from datetime import datetime
import torch.nn.functional as F
from eval_metric import compute_metrics
from losses import skewed_forward_kl, skewed_reverse_kl, js_distance, tv_distance, forward_kl, reverse_kl
from mixture_of_experts import count_parameters

writer = None


def parse_args():
    global writer
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--template', type=int, default=1, help="1--> instruct tuning, 2--> text summary, 3--> question answering, 4--> machine translation")
    parser.add_argument('--data_cache_path', type=str, default=None, )
    parser.add_argument("--model_name_or_path", type=str, help="Path to pretrained model or model identifier from huggingface.co/models.", required=True, )
    # train
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.", )
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help="Batch size (per device) for the evaluation dataloader.", )
    parser.add_argument("--max_seq_len", type=int, default=512, help="The maximum sequence length.", )
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate (after the potential warmup period) to use.", )
    parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.", )
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="cosine", help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"], )
    parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the model.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Enable HF gradient checkpointing for model.')
    parser.add_argument("--dropout", type=float, default=None, help="If dropout configured, use it. "     "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload', action='store_true', help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype', type=str, default='fp16', choices=['fp16', 'bf16', 'fp32'], help='Training data type')
    parser.add_argument('--zero_stage', type=int, default=0, help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim", type=int, default=0, help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--lora_module_name", type=str, default="decoder.layers.", help="split by ',' ")
    parser.add_argument("--lora_learning_rate", type=float, default=5e-4,
                        )
    ## low precision
    parser.add_argument('--compute_fp32_loss', action='store_true', )
    ## Print loss
    parser.add_argument('--print_loss_step', type=int, default=10)
    # evaluation
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--eval_interval', type=int, default=-1)
    parser.add_argument('--eval_num', type=int, default=1000)
    parser.add_argument('--data_reload', action='store_true')
    parser.add_argument('--dont_eval_begin', action='store_true')
    # generation
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    # labels control
    parser.add_argument("--only_resp_loss", action='store_true')
    # moe
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--num_experts_per_tok", type=int, default=2)
    parser.add_argument("--mlp_lora_alpha", type=float, default=1)
    parser.add_argument("--mlp_lora_r", type=int, default=64)
    parser.add_argument("--moe_layer_ids", type=str, default='')
    parser.add_argument("--nums_of_experts", type=str, default='')
    parser.add_argument("--train_mode", type=str, default='full', choices=['full', 'only_moe'])
    parser.add_argument("--load_from_moe", action="store_true")
    # teacher model
    parser.add_argument('--teacher_model_path', type=str)
    # distillation
    parser.add_argument('--kd', action='store_true', help="If true, do knowledge distillation, else do sft")
    parser.add_argument('--skew_alpha', type=float, default=0.1)
    parser.add_argument('--lm_loss_ratio', type=float, default=0.0)
    parser.add_argument('--kd_loss_type', type=str, default='srkl', choices=['srkl', 'sfkl', ])
    # save log
    parser.add_argument('--save_log', action='store_true')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args.add_eot_token = False
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tb_log"))
    return args


def save_args(args):
    target_dir = os.path.join(args.output_dir, 'experiment_args')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(target_dir, 'args.json'), 'w') as f:
        dict_args = vars(args)
        dict_args['datetime'] = f"{formatted_datetime}"
        json.dump(dict_args, f, indent=2)


def color_print(text, color='blue', flush=False):
    print(colored(text, color), flush=flush)


def get_distil_loss(args, labels, logits, teacher_logits):
    if "sfkl" in args.kd_loss_type:
        distil_loss = skewed_forward_kl(logits, teacher_logits, labels, lam=args.skew_alpha)
    elif "srkl" in args.kd_loss_type:
        distil_loss = skewed_reverse_kl(logits, teacher_logits, labels, lam=args.skew_alpha)
    elif "jsd" in args.kd_loss_type:
        distil_loss = js_distance(logits, teacher_logits, labels)
    elif "tvd" in args.kd_loss_type:
        distil_loss = tv_distance(logits, teacher_logits, labels)
    elif "fkl" in args.kd_loss_type or args.kd_loss_type == "kd":
        distil_loss = forward_kl(logits, teacher_logits, labels)
    elif "rkl" in args.kd_loss_type:
        distil_loss = reverse_kl(logits, teacher_logits, labels)
    else:
        raise NotImplementedError
    return distil_loss


def evaluation_gene(model, tokenizer, eval_dataloader, device, current_step, args):
    model.eval()
    if args.do_sample:
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=6,
            max_length=512,
            min_length=None,
        )
    else:
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    all_generate_ids = []
    all_resp_ids = []
    all_prompt_ids = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v for k, v in batch.items() if k in ['prompt_ids', 'prompt_mask', 'resp_ids']}
        batch = to_device(batch, device)

        all_resp_ids.append(batch['resp_ids'])
        all_prompt_ids.append(batch['prompt_ids'])
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=batch['prompt_ids'],
                attention_mask=batch['prompt_mask'],
                generation_config=generation_config,
                use_cache=True,
            )

            full_ids = gen_out
            full_ids = F.pad(
                full_ids,
                (0, args.max_seq_len - full_ids.shape[1]),
                value=tokenizer.pad_token_id,
            )

            generate_ids = full_ids[:, batch['prompt_ids'].size(1):]
            all_generate_ids.append(generate_ids)
    all_generate_ids = torch.cat(all_generate_ids, dim=0)
    all_resp_ids = torch.cat(all_resp_ids, dim=0)
    all_prompt_ids = torch.cat(all_prompt_ids, dim=0)
    generation_resp = tokenizer.batch_decode(all_generate_ids, skip_special_tokens=True)
    reference_resp = tokenizer.batch_decode(all_resp_ids, skip_special_tokens=True)
    prompts = tokenizer.batch_decode(all_prompt_ids, skip_special_tokens=True)
    zh_bleu = True if args.template == 4 else False
    res = compute_metrics(generation_resp, [[x] for x in reference_resp], zh_bleu=zh_bleu)

    if args.global_rank == 0:
        writer.add_scalar('rougeL', res['rougeL'], current_step)
        writer.add_scalar('exact_match', res['exact_match'], current_step)
        writer.add_scalar('bleu', res['bleu'], current_step)
        writer.flush()
    res['eval_num'] = len(generation_resp)
    eval_record = []
    for g, r, p in zip(generation_resp, reference_resp, prompts):
        eval_record.append({"generation_resp": g, 'reference_resp': r, "prompt": p})
    target_dir = os.path.join(args.output_dir, f'eval/step_{current_step}')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    with open(os.path.join(target_dir, 'generation.json'), 'w') as f:
        json.dump(eval_record, f, ensure_ascii=False, indent=2)
    with open(os.path.join(target_dir, f'performance.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    model.train()
    return res


def evaluation_loss(model, eval_dataloader, device, current_steps, args):
    model.eval()
    # eval loss
    losses = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        if 'labels' not in batch:
            batch['labels'] = batch['input_ids']
        batch = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
        batch = to_device(batch, device)
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        losses += loss.float()
    losses = losses / (step + 1)
    try:
        perplexity = torch.exp(losses).item()
    except OverflowError:
        perplexity = float("inf")
    model.train()
    if args.global_rank == 0:
        writer.add_scalar('perplexity', perplexity, current_steps)
        writer.add_scalar('eval_loss', losses.item(), current_steps)
        writer.flush()
    return perplexity, losses.item()


def train_sft(args, model, tokenizer, train_dataloader, eval_dataloader, device, lr_scheduler):
    color_print("**************** [train_sft] ****************", 'green')
    current_steps = 0

    if args.do_train:
        for epoch in range(args.num_train_epochs):
            print(f"|---> Beginning of Epoch [{epoch + 1}/{args.num_train_epochs}], Total Micro Batches {len(train_dataloader)}")
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader)):
                micro_beg_time = time.time()
                batch = to_device(batch, device)
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], use_cache=False)
                loss = outputs.loss
                model.backward(loss)
                model.step()
                current_steps += 1

                if args.global_rank == 0:
                    writer.add_scalar('loss', loss.item(), current_steps)
                    writer.flush()
                if current_steps % args.print_loss_step == 0:
                    micro_step_time = time.time() - micro_beg_time
                    print(f"Epoch: [{epoch + 1}/{args.num_train_epochs}] | Step: {current_steps} | loss = {loss:.6f} | lr = {lr_scheduler.get_last_lr()[0]:.4e} | "
                          f"micro_step_time = {micro_step_time:.4f} s", flush=True)
                    if current_steps % (10 * args.print_loss_step) == 0:
                        color_print(f"{args.output_dir}", 'yellow')

                if args.moe and (current_steps % (50 * args.gradient_accumulation_steps) == 0):
                    torch.cuda.empty_cache()

                if (current_steps / args.gradient_accumulation_steps % args.eval_interval == 0):
                    if args.moe: torch.cuda.empty_cache()
                    # Evaluate perplexity on the validation set.
                    color_print("-----------------------------------------------")
                    color_print(f"|---> Evaluating ... Epoch: [{epoch + 1}/{args.num_train_epochs}]")
                    res = evaluation_gene(model, tokenizer, eval_dataloader, device, current_steps, args)
                    color_print(f"|---> eval generation: {res}", 'magenta')
                    perplexity, eval_loss = evaluation_loss(model, eval_dataloader, device, current_steps, args)
                    color_print(f"|---> eval ppl: {perplexity:.4f}, loss: {eval_loss:.4f}", )
                    color_print("-----------------------------------------------")

                    # save ckpt
                    if args.global_rank == 0 and args.lora_dim == 0:
                        if current_steps == args.total_train_steps:
                            save_hf_format(model, tokenizer, args.output_dir, sub_folder=f"ckpt/final_{current_steps}_ppl_{perplexity:.2f}")
                        else:
                            save_hf_format(model, tokenizer, args.output_dir, sub_folder=f"ckpt/step_{current_steps}_ppl_{perplexity:.2f}")
                    if args.global_rank == 0 and args.lora_dim > 0 and (not current_steps == args.total_train_steps):
                        # save adapter
                        model.save_pretrained(os.path.join(args.output_dir, f"ckpt/adapter_step_{current_steps}_ppl_{perplexity:.2f}"))
                    if args.global_rank == 0 and args.lora_dim > 0 and (current_steps == args.total_train_steps):
                        model.merge_and_unload()
                        save_hf_format(model, tokenizer, args.output_dir, sub_folder=f"ckpt/final_{current_steps}_ppl_{perplexity:.2f}")


def train_kd(args, teacher_model, model, tokenizer, train_dataloader, eval_dataloader, device, lr_scheduler):
    teacher_model.eval()
    color_print("**************** [train_kd] ****************", 'green')
    current_steps = 0

    if args.do_train:
        for epoch in range(args.num_train_epochs):
            print(f"|---> Beginning of Epoch [{epoch + 1}/{args.num_train_epochs}], Total Micro Batches {len(train_dataloader)}")
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader)):
                micro_beg_time = time.time()
                batch = to_device(batch, device)
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], use_cache=False)
                logits = outputs.logits
                lm_loss = outputs.loss
                with torch.no_grad():
                    teacher_logits = teacher_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
                # color_print(f"logits.shape={logits.shape}, teacher_logits.shape={teacher_logits.shape}")
                distill_loss = get_distil_loss(args, batch['labels'], logits, teacher_logits)
                loss = args.lm_loss_ratio * lm_loss + distill_loss
                model.backward(loss)
                model.step()
                current_steps += 1

                if args.global_rank == 0:
                    writer.add_scalar('lm_loss', lm_loss.item(), current_steps)
                    writer.add_scalar('distill_loss', distill_loss.item(), current_steps)
                    writer.add_scalar('loss', loss.item(), current_steps)
                    writer.flush()
                if current_steps % args.print_loss_step == 0:
                    micro_step_time = time.time() - micro_beg_time
                    print(f"Epoch: [{epoch + 1}/{args.num_train_epochs}] | Step: {current_steps} | loss = {loss:.6f} | lr = {lr_scheduler.get_last_lr()[0]:.4e} | "
                          f"micro_step_time = {micro_step_time:.4f} s", flush=True)
                    if current_steps % (10 * args.print_loss_step) == 0:
                        color_print(f"{args.output_dir}", 'yellow')

                if args.moe and (current_steps % (50 * args.gradient_accumulation_steps) == 0):
                    torch.cuda.empty_cache()

                if (current_steps / args.gradient_accumulation_steps % args.eval_interval == 0):
                    if args.moe: torch.cuda.empty_cache()
                    # Evaluate perplexity on the validation set.
                    color_print("-----------------------------------------------")
                    color_print(f"|---> Evaluating ... Epoch: [{epoch + 1}/{args.num_train_epochs}]")
                    res = evaluation_gene(model, tokenizer, eval_dataloader, device, current_steps, args)
                    color_print(f"|---> eval generation: {res}", 'magenta')
                    perplexity, eval_loss = evaluation_loss(model, eval_dataloader, device, current_steps, args)
                    color_print(f"|---> eval ppl: {perplexity:.4f}, loss: {eval_loss:.4f}", )
                    color_print("-----------------------------------------------")

                    # save ckpt
                    if args.global_rank == 0 and (not current_steps == args.total_train_steps):
                        save_hf_format(model, tokenizer, args.output_dir, sub_folder=f"ckpt/step_{current_steps}_ppl_{perplexity:.2f}")
                    if args.global_rank == 0 and (current_steps == args.total_train_steps):
                        model = model.merge_and_unload()
                        save_hf_format(model, tokenizer, args.output_dir, sub_folder=f"ckpt/final_{current_steps}_ppl_{perplexity:.2f}")


def load_teacher_model(args):
    color_print("Loading teacher model ...")
    beg_time = time.time()
    teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model_path, torch_dtype=torch.float32, device_map='auto')
    teacher_model = teacher_model.half().cuda()
    # teacher_model = teacher_model.cuda()
    color_print(f"Loading teacher cost time: {time.time() - beg_time:.2f} s")
    return teacher_model


def get_lora_model(model, args):
    from peft import LoraConfig, TaskType
    # transformers.GPT2LMHeadModel
    lora_config = LoraConfig(
        r=args.lora_dim,
        target_modules=args.lora_module_name.split(','),
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.00
    )
    from peft import get_peft_model
    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    # color_print(lora_model)
    return lora_model


def make_gradient_checkpointing(model, args):
    def make_inputs_require_grad(module, _input, output):
        output.requires_grad_(True)

    if args.lora_dim == 0:
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    else:
        if args.gradient_checkpointing:
            model.base_model.gradient_checkpointing_enable()
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        else:
            model.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    save_args(args)

    # save log
    if args.save_log:
        os.makedirs(os.path.join(args.output_dir, f"logs"), exist_ok=True)
        old_stdout = sys.stdout
        # 重定向标准输出到 out.txt
        sys.stdout = open(os.path.join(args.output_dir, f"logs/output.txt"), "w")

    ds_config = get_train_ds_config(offload=args.offload, dtype=args.dtype, stage=args.zero_stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

    set_random_seed(args.seed)
    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    if args.moe:
        from mixture_of_experts import create_moe_lora_model
        model = create_moe_lora_model(model_path=args.model_name_or_path,
                                      num_experts_per_tok=args.num_experts_per_tok,
                                      mlp_lora_alpha=args.mlp_lora_alpha,
                                      mlp_lora_r=args.mlp_lora_r,
                                      moe_layer_ids=[int(x) for x in args.moe_layer_ids.split(',') if x != ""],
                                      nums_of_experts=[int(x) for x in args.nums_of_experts.split(',') if x != ""],
                                      train_mode=args.train_mode,
                                      load_from_moe=args.load_from_moe
                                      )
    else:
        model = create_hf_model(AutoModelForCausalLM, args.model_name_or_path, tokenizer, ds_config, dropout=args.dropout)

    if args.lora_dim > 0:
        model = get_lora_model(model, args)

    make_gradient_checkpointing(model, args)

    # Prepare the data
    train_dataset, eval_dataset = create_sft_dataset(args.data_path, args.max_seq_len, tokenizer,
                                                     args.data_cache_path, args.eval_num, args.only_resp_loss, template=args.template, reload=args.data_reload)

    # DataLoaders creation:
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)

    if args.do_train:
        # Split weights in two groups, one with weight decay and the other not.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, args.weight_decay, args.lora_learning_rate)
        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.total_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                     num_training_steps=args.num_train_epochs * num_update_steps_per_epoch)
        model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, args=args, config=ds_config, lr_scheduler=lr_scheduler, dist_init_required=True)

    if args.eval_interval == -1:
        args.eval_interval = len(train_dataloader) // args.gradient_accumulation_steps

    color_print(model, 'yellow')
    if args.do_train:
        for n, p in model.named_parameters():
            if p.requires_grad: color_print(f"name: {n} ---> {p.requires_grad}")
    count_parameters(model)

    if args.do_eval and not args.dont_eval_begin:
        current_steps = 0
        color_print("-----------------------------------------------")
        color_print(f"|---> Evaluating ... ")
        res = evaluation_gene(model, tokenizer, eval_dataloader, device, current_steps, args)
        color_print(f"|---> eval generation: {res}", 'magenta')
        perplexity, eval_loss = evaluation_loss(model, eval_dataloader, device, current_steps, args)
        color_print(f"|---> eval ppl: {perplexity:.4f}, loss: {eval_loss:.4f}", )
        color_print("-----------------------------------------------")

    if args.do_train:
        if args.kd:
            # knowledge distillation
            teacher_model = load_teacher_model(args)
            train_kd(args, teacher_model, model, tokenizer, train_dataloader, eval_dataloader, device, lr_scheduler)
        else:
            # supervised finetune
            train_sft(args, model, tokenizer, train_dataloader, eval_dataloader, device, lr_scheduler)

    # save log
    if args.save_log:
        sys.stdout.close()
        sys.stdout = old_stdout


if __name__ == "__main__":
    main()
