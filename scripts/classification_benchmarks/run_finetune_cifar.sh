python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/cifar/resnet50_finetune_cifar.yaml \
    --output /data/train_log_NEW/resnet50_finetune_cifar100_byol -j 8