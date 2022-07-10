python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/flowers/resnet50_finetune_flowers.yaml \
    --output /data/train_log_NEW/resnet50_finetune_flowers_byol -j 8