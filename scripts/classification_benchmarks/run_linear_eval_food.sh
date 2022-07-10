python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/food101/resnet50_linear_eval_food.yaml \
    --output /data/train_log_NEW/resnet50_linear_eval5w5f -j 8