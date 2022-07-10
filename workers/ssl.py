import os
import random
import importlib
import time

import torch
import torch.backends.cudnn as cudnn
from timm.optim import create_optimizer

from config import get_config, update_config
import datasets
import models
from losses import LOSS
import quant_tools
import utils


def ssl(args):
    config = get_config(args)
    config.defrost()
    device = torch.device(config.DEVICE)

    # ---- setup logger and output ----
    os.makedirs(args.output, exist_ok=True)
    logger = utils.train.construct_logger(config.SSL.TYPE, config.OUTPUT)

    if config.TRAIN.USE_DDP: 
        utils.ddp.init_distributed_mode(config)

    cudnn.benchmark = True
    utils.train.set_random_seed(config)

    # build dataloaders
    train_preprocess  = datasets.build_ssl_transform(config, two_crop=True)
    print(train_preprocess)
    train_dataloader = datasets.build_dataloader("train", config, train_preprocess)
    
    # SSL wrapper for model,
    print('build {} model'.format(config.SSL.TYPE))
    model = utils.builder.build_ssl_model(models.__dict__[config.MODEL.ARCH], config)
    print(model)

    if config.MODEL.CHECKPOINT:
        ckpt = torch.load(config.MODEL.CHECKPOINT, map_location="cpu")
        config.defrost()
        config.TRAIN.START_EPOCH = ckpt['epoch']
        state_dict = ckpt['state_dict']
        print(state_dict.keys())
        print('='*20)
        print(model.state_dict().keys())
        print('='*20)
        msg = model.load_state_dict(state_dict, strict=False)
        print('state dict missing', set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(config.MODEL.CHECKPOINT))

    if config.TRAIN.SYNC_BN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)
    print(model)

    if config.TRAIN.USE_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model.to(device),
            device_ids=[config.LOCAL_RANK],
            find_unused_parameters=True,
        )
    utils.logging_information(logger, config, str(model))

    runner_class = importlib.import_module("runners." + config.TRAIN.RUNNER.NAME).RUNNER
    train_runner = runner_class(
        mode="train",
        model=model,
        config=config,
        dataloader=train_dataloader,
        logger=logger,
        mixup_fn=None,
    )

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if config.TRAIN.USE_DDP:
            train_dataloader.sampler.set_epoch(epoch)
        train_runner(model, epoch)
