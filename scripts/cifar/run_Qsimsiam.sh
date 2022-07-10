#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
#    ssl --config ./configs/cifar/resnet18_ssql_simsiam_cifar.yaml \
#    --output /data/train_log_SSQL_release/cifar10/r18/SSQL_SimSiam_w_2_8_f_4_8_cifar_r18_cifar10 -j 8

#for resnet34
#python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
#    ssl --config ./configs/cifar/resnet34_ssql_simsiam_cifar.yaml \
#    --output /data/train_log_SSQL_release/cifar10/r34/SSQL_SimSiam_w_2_8_f_4_8_cifar_r34_cifar10 -j 8


#for resnet50, 8 gpus
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    ssl --config ./configs/cifar/resnet50_ssql_simsiam_cifar.yaml \
    --output /data/train_log_SSQL_release/cifar10/r50/SSQL_SimSiam_w_2_8_f_4_8_cifar_r50_cifar10 -j 8