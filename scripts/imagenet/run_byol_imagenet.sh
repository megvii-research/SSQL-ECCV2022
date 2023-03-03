python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    ssl --config ./configs/imagenet/resnet18_byol_imagenet.yaml \
    --output ./train_log/imagenet/r18/byol_r18_imagenet -j 8
