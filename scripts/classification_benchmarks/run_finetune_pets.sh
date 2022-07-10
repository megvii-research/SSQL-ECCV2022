python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/pets/resnet18_finetune_pets.yaml \
    --output /data/train_log_NEW/resnet18_finetune_pets -j 8