python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/flowers/resnet50_linear_eval_flowers.yaml \
    --output /data/train_log_NEW/resnet50_linear_eval_flowers_mocov2 -j 8