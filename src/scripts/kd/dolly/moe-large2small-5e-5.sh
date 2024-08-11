DATETIME=$(date '+%Y%m%d_%H%M%S')
ROOT_DIR=/the/path/to/MoE-KD

MODEL_PATH=${ROOT_DIR}/results/sft/gpt2/moe_small_[lr5e-4]_20240318_190403/ckpt/[24.19]step_4000_ppl_48.15
TEACHER_PATH=${ROOT_DIR}/results/sft/gpt2/large_[lr5e-5]_20240319_022443/ckpt/[26.94]_step_18400_ppl_18.16
DATA_PATH=${ROOT_DIR}/dataset/dolly
OUTPUT=${ROOT_DIR}/results/kd/gpt2/large2small_[lr5e-5]_moe${DATETIME}
BATCH_SIZE=8

OPTS=""
OPTS+=" --kd"
OPTS+=" --kd_loss_type srkl"
OPTS+=" --lm_loss_ratio 0"
OPTS+=" --moe"
OPTS+=" --load_from_moe"
OPTS+=" --dont_eval_begin"

OPTS+=" --do_train"
OPTS+=" --do_eval"
OPTS+=" --data_path ${DATA_PATH}"
OPTS+=" --model_name_or_path ${MODEL_PATH}"
OPTS+=" --teacher_model_path ${TEACHER_PATH}"
OPTS+=" --per_device_train_batch_size ${BATCH_SIZE}"
OPTS+=" --per_device_eval_batch_size ${BATCH_SIZE}"
OPTS+=" --gradient_accumulation_steps 2"
OPTS+=" --learning_rate 5e-5"
OPTS+=" --num_train_epochs 5"
OPTS+=" --lr_scheduler_type cosine"
OPTS+=" --weight_decay 0.01"
OPTS+=" --num_warmup_steps 50"
OPTS+=" --output_dir ${OUTPUT}"
OPTS+=" --gradient_checkpointing "
OPTS+=" --dtype fp16"
OPTS+=" --print_loss_step 5"
OPTS+=" --eval_num 500"
#OPTS+=" --data_reload"
OPTS+=" --eval_interval 200"
OPTS+=" --only_resp_loss"
OPTS+=" --seed 10"

# eval generation
OPTS+=" --do_sample"
OPTS+=" --temperature 0.5"
OPTS+=" --max_new_tokens 256"



deepspeed train_main.py ${OPTS}



