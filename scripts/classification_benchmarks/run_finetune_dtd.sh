python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/dtd/resnet50_finetune_dtd.yaml \
    --output /data/train_log_NEW/resnet50_finetune_dtd -j 8