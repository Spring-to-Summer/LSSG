export PYTHONPATH=.

torchrun --nproc_per_node=4 --master_port=6001 tools/play_llm_game.py \
    --taboo_max_turns 15 \
    --attacker_model_name_or_path "./ckpts/im/checkpoint-6889" \
    --defender_model_name_or_path "./ckpts/im/checkpoint-6889" \
    --model_prefix "im_llama2" \
    --data_path "./data/filtered_items.txt" \
    --output_dir "./data/self_play_results" \
    --per_device_eval_batch_size 1 \
    --task_type "sampling" \
    --data_suffix "all_words" \
    --max_length 2048 \
    --max_new_tokens 256 \
    --logging_steps 5 \
    --bf16 True \
    --tf32 True