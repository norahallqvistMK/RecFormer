python finetune_classification.py \
    --pretrain_ckpt pretrain_ckpt/fraud_pretrain_ckpt.bin \
    --data_path transactional_data_process/classification_data \
    --num_train_epochs 20 \
    --batch_size 1 \
    --device 1 \
    --fp16 \
    --fix_word_embedding