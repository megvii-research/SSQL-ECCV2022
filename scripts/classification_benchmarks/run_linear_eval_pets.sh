python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/pets/resnet50_linear_eval_pets.yaml \
    --output /data/train_log_NEW/resnet18_linear_eval5w5f -j 8