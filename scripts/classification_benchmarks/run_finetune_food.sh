python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/food101/resnet50_finetune_food.yaml \
    --output ./train_log_NEW/resnet50_finetune_food+floatloss -j 8