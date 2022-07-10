# SSQL-ECCV2022
Official code for **S**ynergistic **S**elf-supervised and **Q**uantization **L**earning (Accepted to ECCV 2022 **oral presentation**)

<div align=center><img src="https://github.com/CupidJay/SSQL-ECCV2022/blob/main/method.png" width="75%"></div>

## Introduction
In this paper, we propose a method called
synergistic self-supervised and quantization learning (SSQL) to pretrain
quantization-friendly self-supervised models facilitating downstream deployment.
SSQL contrasts the features of the quantized and full precision
models in a self-supervised fashion, where the bit-width for the quantized
model is randomly selected in each step. SSQL not only significantly improves
the accuracy when quantized to lower bit-widths, but also boosts
the accuracy of full precision models in most cases. By only training once,
SSQL can then benefit various downstream tasks at different bit-widths
simultaneously. Moreover, the bit-width flexibility is achieved without
additional storage overhead, requiring only one copy of weights during
training and inference. We theoretically analyze the optimization process
of SSQL, and conduct exhaustive experiments on various benchmarks to
further demonstrate the effectiveness of our method.

## Getting Started

### Prerequisites
* python 3
* PyTorch (= 1.10.0)
* torchvision (= 0.11.1)
* Numpy
* CUDA 10.1

### CIFAR Experiments
- Pre-training stage using SSQL (c.f. [scripts/cifar/run_Qsimsiam.sh](scripts/cifar/run_Qsimsiam.sh)), run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    ssl --config ./configs/cifar/resnet18_ssql_simsiam_cifar.yaml \
    --output [your_checkpoint_dir] -j 8
```
You can set hyper-parameters manually in the corresponding .yaml file (e.g., [configs/cifar/resnet18_ssql_simsiam_cifar.yaml](configs/cifar/resnet18_ssql_simsiam_cifar.yaml) here).
```python
TRAIN:
    EPOCHS: 400 # Total training epochs
    DATASET: cifar10 # Specify datasets
    BATCH_SIZE: 128 # batch-size for each gpu
    OPTIMIZER: 
        NAME: sgd
        MOMENTUM: 0.9
        WEIGHT_DECAY: 0.0005
    LR_SCHEDULER:
        WARMUP_EPOCHS: 0
        BASE_LR: 0.05  
        MIN_LR: 0.
        TYPE: cosine # lr scheduler
    LOSS: 
        CRITERION:
            NAME: CosineSimilarity # loss function
QUANT:
    W:
        BIT_RANGE: [2, 9] # the bit range for weights in SSQL
    A:
        BIT_RANGE: [4, 9] # the bit range for activation in SSQL
SSL:
    TYPE: SSQL_SimSiam # SSL algorithm type
    SETTING:
        DIM: 2048 # dimension for the output feature
        HIDDEN_DIM: 2048 # dimension for the hidden MLP
```

- You can also use other baseline methods for pre-training. For example, you can run [scripts/cifar/run_simsiam.sh](scripts/cifar/run_simsiam.sh) for SimSiam, [scripts/cifar/run_moco.sh](scripts/cifar/run_moco.sh) for MoCo.

- Linear evaluation stage using pre-trained models (c.f. [scripts/cifar/run_linear_eval.sh](scripts/cifar/run_linear_eval.sh)), run:
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    linear  --config ./configs/cifar/resnet18_linear_eval_cifar.yaml \
    --output [output_dir] -j 8
```
To specify the pre-trained model path and evaluation bit-width, you need to mannually modify the .yaml file ([configs/cifar/resnet18_linear_eval_cifar.yaml](./configs/cifar/resnet18_linear_eval_cifar.yaml)). For instance, our pretrained model is ./checkpoint.pth.tar and we want to evaluate it with weight and activation both quantized to 4 bits (i.e., 4w4f), the modification in yaml file should look like
```python
MODEL:
    PRETRAINED: ./checkpoint.pth.tar # you need to change here
QUANT:
    TYPE: ptq
    W:
        BIT: 4 # you need to change here
        SYMMETRY: True
        QUANTIZER: uniform
        GRANULARITY : channelwise
        OBSERVER_METHOD:
            NAME: MINMAX
    A:
        BIT: 4 # you need to change here
        SYMMETRY: False
        QUANTIZER: uniform
        GRANULARITY : layerwise
        OBSERVER_METHOD:
            NAME: MINMAX
```

### ImageNet Experiments

#### Pre-training and Linear Evaluation
- Pre-training stage using SSQL (c.f. [scripts/imagenet/run_Qsimsiam_imagenet.sh](scripts/imagenet/run_Qsimsiam_imagenet.sh)), run:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    ssl --config ./configs/imagenet/resnet18_ssql_simsiam_imagenet.yaml \
    --output [your_checkpoint_dir] -j 8
```

- Linear evaluation on ImageNet (c.f. [scripts/imagenet/run_linear_eval_imagenet.sh](scripts/imagenet/run_linear_eval_imagenet.sh)). If you want to specify the model weights and quantization bits, see the instructions on CIFAR above and modify the corresponding [configs/ssl_qat/resnet18_linear_eval_imagenet.yaml](configs/ssl_qat/resnet18_linear_eval_imagenet.yaml).

#### Transferring Experiments
[To be updated]

## Citation
Please consider citing our work in your publications if it helps your research.
```
@article{SSQL,
   title         = {Synergistic Self-supervised and Quantization Learning},
   author        = {Yun-Hao Cao, Yechang Huang, Peiqin Sun, Jianxin Wu and Shuchang Zhou},
   year          = {2022},
   booktitle = {The European Conference on Computer Vision}}
```

