# Use distributed data parallel
python lightning_pretrain.py \
    --model_name_or_path allenai/longformer-base-4096 \
    --train_file transactional_data_process/pretrain_data/train.json \
    --dev_file transactional_data_process/pretrain_data/train.json \
    --item_attr_file transactional_data_process/pretrain_data/meta_data.json \
    --output_dir result/recformer_pretraining \
    --num_train_epochs 20 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8  \
    --batch_size 1 \
    --learning_rate 5e-5 \
    --temp 0.05 \
    --device 1 \
    --fp16 \
    --fix_word_embedding