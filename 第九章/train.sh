accelerate launch --config_file=accelerate.yaml deepspeed_train.py \
  --val_dataset_path="Belle_val.json" \
--val_dataset_cache_path="Belle_val.h5" \
--train_dataset_cache_path="Belle_train.h5" \
--train_dataset_path="Belle_train.json" \
--model_path="../BigFile/Llama3-8B-pretrain-weight" \
--config_name="../BigFile/Llama3-8B-pretrain-weight" \
--tokenizer_path="../BigFile/Llama3-8B-pretrain-weight" \
--per_device_train_batch_size="2" \
--per_device_eval_batch_size="2" \
--learning_rate="2e-5" \
--num_train_epochs="1" \
--max_train_steps="10000" \
--gradient_accumulation_steps="2" \
--lr_scheduler_type="constant_with_warmup" \
--output_dir="./saved" \
--seed="42" \
--max_length="512" \
--checkpointing_steps=2000
