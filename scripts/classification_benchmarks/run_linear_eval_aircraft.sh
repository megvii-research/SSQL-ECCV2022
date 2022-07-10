python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/aircraft/resnet18_linear_eval_aircraft.yaml \
    --output /data/train_log_NEW/resnet18_linear_eval_aircraft_simsiam -j 8