python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/aircraft/resnet50_finetune_aircraft.yaml \
    --output /data/train_log_NEW/resnet50_finetune_aircraft -j 8