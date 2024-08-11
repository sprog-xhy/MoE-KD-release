import os

ROOT_DIR = "/the/path/to/MoE-KD"
DATA_PATH = f"{ROOT_DIR}/dataset/dolly"
MODEL_PATH = f"{ROOT_DIR}/results/sft/dolly/small_[lr5e-4]_20240318_160959/ckpt/step_4600_ppl_48.69"
TEACHER_PATH = f"{ROOT_DIR}/results/sft/dolly/large_[lr5e-5]_20240319_022443/ckpt/[26.94]_step_18400_ppl_18.16"


def get_fixed_args():
    OPTS = ""
    OPTS += f" --kd"
    OPTS += f" --save_log"
    OPTS += f" --kd_loss_type srkl"
    OPTS += f" --lm_loss_ratio 0.1"
    OPTS += f" --dont_eval_begin"
    OPTS += f" --do_train"
    OPTS += f" --do_eval"
    OPTS += f" --data_path {DATA_PATH}"
    OPTS += f" --model_name_or_path {MODEL_PATH}"
    OPTS += f" --teacher_model_path {TEACHER_PATH}"
    OPTS += f" --per_device_train_batch_size 8"
    OPTS += f" --per_device_eval_batch_size 8"
    OPTS += f" --gradient_accumulation_steps 2"
    OPTS += f" --learning_rate 5e-5"
    OPTS += f" --num_train_epochs 3"
    OPTS += f" --lr_scheduler_type cosine"
    OPTS += f" --weight_decay 0.01"
    OPTS += f" --num_warmup_steps 50"
    OPTS += f" --gradient_checkpointing "
    OPTS += f" --dtype fp16"
    OPTS += f" --print_loss_step 10"
    OPTS += f" --eval_num 500"
    # OPTS+=f" --data_reload"
    OPTS += f" --eval_interval 200"
    OPTS += f" --only_resp_loss"
    OPTS += f" --seed 10"
    # eval gfeneration
    OPTS += f" --do_sample"
    OPTS += f" --temperature 0.5"
    OPTS += f" --max_new_tokens 256"

    return OPTS


def get_moe_fixed_args():
    OPTS = ""
    OPTS += f" --moe"
    OPTS += f" --num_experts_per_tok 2"
    OPTS += f" --train_mode full"
    return OPTS


def get_modified_args(mlp_lora_r=16, moe_layer_ids='6,7,8,9,10,11', nums_of_experts='2,2,4,4,8,8'):
    OPTS = ""
    OPTS += f" --mlp_lora_r {mlp_lora_r}"
    OPTS += f" --moe_layer_ids {moe_layer_ids}"
    OPTS += f" --nums_of_experts {nums_of_experts}"

    output = f"{ROOT_DIR}/results/config_expert/dolly/small_[r{mlp_lora_r}]_[l{moe_layer_ids}]_[e{nums_of_experts}]"
    OPTS += f" --output_dir {output}"
    print("-----------------------------------")
    print(f"{OPTS}")
    print("-----------------------------------")
    return OPTS


def run_experiment(opts):
    os.system(f"deepspeed train_main.py {opts}")


def exp1():
    # mlp_lora_r
    for mlp_lora_r in [4, 16, 64,256]:
        opts = get_fixed_args() + get_moe_fixed_args() + get_modified_args(mlp_lora_r=mlp_lora_r)
        run_experiment(opts)


def exp2():
    e1 = {'moe_layer_ids': '0,1,2,3,4,5,6,7,8,9,10,11', 'nums_of_experts': '2,2,2,4,4,4,6,6,6,8,8,8'}
    # e2 = {'moe_layer_ids': '6,7,8,9,10,11', 'nums_of_experts': '2,2,4,4,8,8'}
    e3 = {'moe_layer_ids': '9,10,11', 'nums_of_experts': '2,4,8'}
    e4 = {'moe_layer_ids': '11', 'nums_of_experts': '8'}
    for e in [e1, e3, e4]:
        opts = get_fixed_args() + get_moe_fixed_args() + get_modified_args(**e)
        run_experiment(opts)


def exp3():
    for num in [2, 8, 32, 128]:
        opts = get_fixed_args() + get_moe_fixed_args() + get_modified_args(nums_of_experts=','.join([str(num) for _ in range(6)]))
        run_experiment(opts)


if __name__ == "__main__":
    exp1()
    exp2()
    exp3()
