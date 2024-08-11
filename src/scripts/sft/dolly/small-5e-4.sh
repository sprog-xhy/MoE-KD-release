DATETIME=$(date '+%Y%m%d_%H%M%S')
ROOT_DIR=/the/path/to/MoE-KD

MODEL_PATH=${ROOT_DIR}/models/gpt2-small
DATA_PATH=${ROOT_DIR}/dataset/dolly
OUTPUT=${ROOT_DIR}/results/sft/gpt2/small_[lr5e-4]_${DATETIME}
BATCH_SIZE=32

OPTS=""
OPTS+=" --do_train"
OPTS+=" --do_eval"
OPTS+=" --data_path ${DATA_PATH}"
OPTS+=" --model_name_or_path ${MODEL_PATH}"
OPTS+=" --per_device_train_batch_size ${BATCH_SIZE}"
OPTS+=" --per_device_eval_batch_size ${BATCH_SIZE}"
OPTS+=" --gradient_accumulation_steps 1"
OPTS+=" --learning_rate 5e-4"
OPTS+=" --num_train_epochs 10"
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



