python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/resnet18_ssql_moco_cifar.yaml \
    --output ./train_log/cifar10/r18/SSQL_MoCo_w_2_8_f_4_8_cifar_r18_cifar10 -j 8