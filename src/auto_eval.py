import os
from typing import List
from termcolor import colored


def color_print(text, color='blue', flush=False):
    print(colored(text, color), flush=flush)


def get_cmd(data_path, model_path, output_dir, seed=10, eval_batch_size=32, moe=False,eval_num=1000):
    OPTS = ""
    OPTS += " --do_eval"
    OPTS += f" --data_path {data_path}"
    OPTS += f" --model_name_or_path {model_path}"
    OPTS += f" --per_device_eval_batch_size {eval_batch_size}"
    OPTS += f" --output_dir {output_dir}"
    OPTS += " --gradient_checkpointing "
    OPTS += " --dtype fp16"
    OPTS += f" --eval_num {eval_num}"
    # OPTS+=" --data_reload"
    OPTS += f" --seed {seed}"
    # eval generation
    OPTS += " --do_sample"
    OPTS += " --temperature 0.5"
    OPTS += " --max_new_tokens 256"

    if moe:
        OPTS += " --moe"
        OPTS += " --load_from_moe"
        # OPTS += " --num_experts_per_tok 2"
        # OPTS += " --mlp_lora_r 16"
        # OPTS += " --moe_layer_ids 0,1,2,3,4,5,6,7,8,9,10,11"
        # OPTS += " --nums_of_experts 8,8,8,8,8,8,8,8,8,8,8,8"

    return f"deepspeed train_main.py {OPTS}"


def eval_dolly(model_path, output_dir, seed: int | List = 10, eval_batch_size=32, moe=False):
    color_print("------------- [eval_dolly ...] ------------")
    data_path = "../dataset/dolly"
    if isinstance(seed, int):
        seed = [seed]
    for s in seed:
        color_print(f"------------- [eval_dolly --> seed = {s}] ------------")
        cmd = get_cmd(data_path,
                      model_path,
                      os.path.join(output_dir, f'dolly_seed{s}'),
                      s,
                      eval_batch_size,
                      moe)
        os.system(cmd)


def eval_vicuna(model_path, output_dir, seed: int | List = 10, eval_batch_size=32, moe=False):
    color_print("------------- [eval_vicuna ...] ------------")
    data_path = "../dataset/vicuna"
    if isinstance(seed, int):
        seed = [seed]
    for s in seed:
        color_print(f"------------- [eval_vicuna --> seed = {s}] ------------")
        cmd = get_cmd(data_path,
                      model_path,
                      os.path.join(output_dir, f'vicuna_seed{s}'),
                      s,
                      eval_batch_size,
                      moe)
        os.system(cmd)


def eval_self_instruct(model_path, output_dir, seed: int | List = 10, eval_batch_size=32, moe=False):
    color_print("------------- [eval_self_instruct ...] ------------")
    data_path = "../dataset/self-inst"
    if isinstance(seed, int):
        seed = [seed]
    for s in seed:
        color_print(f"------------- [eval_self_instruct --> seed = {s}] ------------")
        cmd = get_cmd(data_path,
                      model_path,
                      os.path.join(output_dir, f'self_instruct_seed{s}'),
                      s,
                      eval_batch_size,
                      moe)
        os.system(cmd)


def eval_super_natural(model_path, output_dir, seed: int | List = 10, eval_batch_size=32, moe=False):
    color_print("------------- [eval_super_natural ...] ------------")
    data_path = "../dataset/super-natural"
    if isinstance(seed, int):
        seed = [seed]
    for s in seed:
        color_print(f"------------- [eval_super_natural --> seed = {s}] ------------")
        cmd = get_cmd(data_path,
                      model_path,
                      os.path.join(output_dir, f'super_natural_seed{s}'),
                      s,
                      eval_batch_size,
                      moe,
                      eval_num=9000)
        os.system(cmd)


def eval_unnatural(model_path, output_dir, seed: int | List = 10, eval_batch_size=32, moe=False):
    color_print("------------- [eval_unnatural ...] ------------")
    data_path = "../dataset/unnatural"
    if isinstance(seed, int):
        seed = [seed]
    for s in seed:
        color_print(f"------------- [eval_unnatural --> seed = {s}] ------------")
        cmd = get_cmd(data_path,
                      model_path,
                      os.path.join(output_dir, f'unnatural_seed{s}'),
                      s,
                      eval_batch_size,
                      moe,
                      eval_num=9000)
        os.system(cmd)


def main(model_path, your_model_name, seed: int | List = 10, eval_batch_size=32, moe=False):
    output_dir = os.path.join("../results/eval", your_model_name)
    eval_dolly(model_path, output_dir, [20,30], eval_batch_size,moe)
    eval_vicuna(model_path, output_dir, [10,20,30], eval_batch_size,moe)
    eval_self_instruct(model_path, output_dir, [10,20,30], eval_batch_size,moe)
    # eval_super_natural(model_path, output_dir, 10, eval_batch_size,moe)
    # eval_unnatural(model_path, output_dir, 10, eval_batch_size,moe)


if __name__ == "__main__":
    main(
        "../results/kd/gpt2/large2small_[lr5e-5]_dense_20240319_202755/ckpt/step_9000_ppl_65.24",
        "kd/gpt2/large2small_[lr5e-5]_dense_20240319_202755/step_9000",
        moe=False
    )
    # main(
    #     "../results/kd/gpt2/large2small_[lr5e-4]_moe_from_pret_20240320_024837/ckpt/step_23448_ppl_40.04",
    #     "kd/gpt2/large2small_[lr5e-4]_moe_from_pret/step_23448",
    #     moe=True
    # )
