python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/ssl_qat/caltech101/resnet18_finetune_caltech.yaml \
    --output /data/train_log_NEW/resnet18_finetune_caltech_simsiam -j 8