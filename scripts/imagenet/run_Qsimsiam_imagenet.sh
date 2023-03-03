python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    ssl --config ./configs/imagenet/resnet18_ssql_simsiam_imagenet.yaml \
    --output ./train_log/imagenet/r18/SSQL_SimSiam_w_2_8_f_4_8_r18_imagenet -j 8
