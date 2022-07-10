python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/resnet18_byol_cifar.yaml \
    --output /data/train_log_SSQL_release/cifar10/r18/byol_cifar_r18_cifar10 -j 8