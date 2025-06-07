python finetune_gpu.py \
    --pretrain_ckpt longformer_ckpt/recformer_seqrec_ckpt.bin \
    --data_path finetune_data_trans/ \
    --num_train_epochs 20 \
    --batch_size 10 \
    --device 1 \
    --fp16 \
    --finetune_negative_sample_size -1 \
    --use_multi_gpu