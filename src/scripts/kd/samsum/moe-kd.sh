DATETIME=$(date '+%Y%m%d_%H%M%S')
ROOT_DIR=/the/path/to/MoE-KD

MODEL_PATH=${ROOT_DIR}/results/sft/samsum/gpt2/small_[lr5e-5]_sft_20240323_010038/ckpt/final_4610_ppl_5.31
TEACHER_PATH=${ROOT_DIR}/results/sft/samsum/gpt2/large_[lr5e-5]_sft_20240321_105624/ckpt/step_6440_ppl_8.57
DATA_PATH=${ROOT_DIR}/dataset/samsum
OUTPUT=${ROOT_DIR}/results/sft/samsum/gpt2/moe_kd_[lr5e-5]_[lm0.1]_${DATETIME}
BATCH_SIZE=8

OPTS=""
OPTS+=" --kd"
OPTS+=" --lm_loss_ratio 0.1"
OPTS+=" --do_train"
OPTS+=" --do_eval"
OPTS+=" --template 2"
OPTS+=" --data_path ${DATA_PATH}"
OPTS+=" --model_name_or_path ${MODEL_PATH}"
OPTS+=" --teacher_model_path ${TEACHER_PATH}"
OPTS+=" --per_device_train_batch_size ${BATCH_SIZE}"
OPTS+=" --per_device_eval_batch_size ${BATCH_SIZE}"
OPTS+=" --gradient_accumulation_steps 2"
OPTS+=" --learning_rate 5e-5"
OPTS+=" --num_train_epochs 3"
OPTS+=" --lr_scheduler_type cosine"
OPTS+=" --weight_decay 0.01"
OPTS+=" --num_warmup_steps 50"
OPTS+=" --output_dir ${OUTPUT}"
OPTS+=" --gradient_checkpointing "
OPTS+=" --dtype fp16"
OPTS+=" --print_loss_step 10"
OPTS+=" --eval_num 4096"
#OPTS+=" --data_reload"
OPTS+=" --eval_interval 500"
OPTS+=" --only_resp_loss"
OPTS+=" --seed 10"

# eval generation
#OPTS+=" --do_sample"
OPTS+=" --temperature 0.5"
OPTS+=" --max_new_tokens 64"
OPTS+=" --max_seq_len 512"

# moe
OPTS+=" --moe"
OPTS+=" --num_experts_per_tok 2"
OPTS+=" --mlp_lora_r 16"
OPTS+=" --moe_layer_ids 6,7,8,9,10,11"
OPTS+=" --nums_of_experts 2,2,4,4,8,8"
OPTS+=" --train_mode full"
#OPTS+=" --load_from_moe"



deepspeed train_main.py ${OPTS}



