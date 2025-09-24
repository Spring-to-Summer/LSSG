MODEL=ckpts/im/checkpoint-6889
OUTPUT_DIR=ckpts/rssng
PREFIX=train_rssng_data_im_llama2

OMP_NUM_THREADS=16 torchrun --nproc_per_node=4 --master_port=6002 train.py \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path $MODEL \
    --ref_model_name_or_path $MODEL \
    --lm_kl_coeff 0.2 \
    --lm_sft_coeff 0.5 \
    --train_method "OfflinePO" \
    --train_data_path "./data/${PREFIX}.json" \
    --remove_unused_columns False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy no \
    --padding_side "right" \
    --truncation_side "left" \
    --max_length 2048 \
    --save_strategy epoch \
    --learning_rate 2e-6 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --weight_decay 0. \
    --deepspeed "./configs/default_offload_opt_param.json" \
    --gradient_checkpointing True \
    --tf32 True \
    --bf16 True
