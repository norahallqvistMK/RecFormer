python finetune.py \
    --pretrain_ckpt longformer_ckpt/recformer_seqrec_ckpt.bin \
    --data_path finetune_data/ \
    --num_train_epochs 1 \
    --batch_size 2 \
    --device 1 \
    --fp16 \
    --finetune_negative_sample_size -1