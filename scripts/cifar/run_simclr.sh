python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/resnet18_simclr_cifar.yaml \
    --output ./train_log/cifar10/r18/simclr_cifar_r18_cifar10 -j 8