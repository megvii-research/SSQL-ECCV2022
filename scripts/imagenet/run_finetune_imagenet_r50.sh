python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/resnet50_finetune_small_imagenet.yaml \
    --output /data/train_log_finetune/small_imagenet/resnet50_RQCL_warmup_fc_0.001_float_imagenet_1_percent_finetune -j 8