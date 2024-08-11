DATETIME=$(date '+%Y%m%d_%H%M%S')
ROOT_DIR=/the/path/to/MoE-KD

MODEL_PATH=${ROOT_DIR}/models/gpt2-small
TEACHER_PATH=${ROOT_DIR}/results/sft/gpt2/large_[lr5e-5]_20240319_022443/ckpt/[26.94]_step_18400_ppl_18.16
DATA_PATH=${ROOT_DIR}/dataset/dolly
OUTPUT=${ROOT_DIR}/results/kd/gpt2/large2small_[lr5e-4]_moe_from_pret_${DATETIME}
BATCH_SIZE=8

OPTS=""
OPTS+=" --kd"
OPTS+=" --kd_loss_type srkl"
OPTS+=" --lm_loss_ratio 0"
OPTS+=" --moe"
OPTS+=" --num_experts_per_tok 2"
OPTS+=" --mlp_lora_r 32"
OPTS+=" --moe_layer_ids 9,10,11"
OPTS+=" --nums_of_experts 2,4,8"
OPTS+=" --train_mode full"
#OPTS+=" --load_from_moe"
OPTS+=" --dont_eval_begin"

OPTS+=" --do_train"
OPTS+=" --do_eval"
OPTS+=" --data_path ${DATA_PATH}"
OPTS+=" --model_name_or_path ${MODEL_PATH}"
OPTS+=" --teacher_model_path ${TEACHER_PATH}"
OPTS+=" --per_device_train_batch_size ${BATCH_SIZE}"
OPTS+=" --per_device_eval_batch_size ${BATCH_SIZE}"
OPTS+=" --gradient_accumulation_steps 4"
OPTS+=" --learning_rate 5e-5"
OPTS+=" --num_train_epochs 20"
OPTS+=" --lr_scheduler_type cosine"
OPTS+=" --weight_decay 0.01"
OPTS+=" --num_warmup_steps 50"
OPTS+=" --output_dir ${OUTPUT}"
OPTS+=" --gradient_checkpointing "
OPTS+=" --dtype fp16"
OPTS+=" --print_loss_step 5"
OPTS+=" --eval_num 500"
#OPTS+=" --data_reload"
OPTS+=" --eval_interval -1"
OPTS+=" --only_resp_loss"
OPTS+=" --seed 10"

# eval generation
OPTS+=" --do_sample"
OPTS+=" --temperature 0.5"
OPTS+=" --max_new_tokens 256"



deepspeed train_main.py ${OPTS}



