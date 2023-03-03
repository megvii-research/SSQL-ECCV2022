python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/resnet18_finetune_small_imagenet.yaml \
    --output ./train_log_finetune/small_imagenet/resnet18_RQCL_float_imagenet_1_percent_finetune -j 8