python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    ssl --config ./configs/cifar/resnet18_ssql_byol_cifar.yaml \
    --output /data/train_log_SSQL_release/cifar10/r18/SSQL_BYOL_w_2_8_f_4_8_cifar_r18_cifar10 -j 8