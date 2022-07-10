python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/resnet18_linear_eval_imagenet.yaml \
    --output /data/train_log_lars/resnet18_byol_4w4f_imagenet_linear_eval -j 8