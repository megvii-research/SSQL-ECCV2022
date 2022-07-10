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
import quant_tools
import utils
import runners

def calibration_run(model, dataloader, device):
    model.eval()
    tmp_size = len(dataloader)
    #self._register_hook(calibration_init_param)
    iter_dataloader = iter(dataloader)
    while tmp_size > 0:
        data = next(iter_dataloader)
        if isinstance(data, (list, tuple)):
            data = data[0]
        with torch.no_grad():
            model(data.to(device))
        tmp_size -= data.shape[0]

def default(args):
    config = get_config(args)
    device = torch.device(config.DEVICE)

    # ---- setup logger and output ----
    os.makedirs(args.output, exist_ok=True)
    logger = utils.train.construct_logger("FLOAT", config.OUTPUT)

    if config.TRAIN.USE_DDP:  # only support single node
        utils.ddp.init_distributed_mode(config)

    cudnn.benchmark = True
    utils.train.set_random_seed(config)

    # build dataloaders
    train_preprocess, eval_preprocess, mixup_fn = datasets.build_transforms(
        config, logger
    )
    train_dataloader = datasets.build_dataloader("train", config, train_preprocess)
    validation_dataloader = datasets.build_dataloader(
        "validation", config, eval_preprocess
    )

    model = models.__dict__[config.MODEL.ARCH]()

    if config.TRAIN.LINEAR_EVAL:
        if config.MODEL.PRETRAINED:
            if 'Qbyol' in config.MODEL.PRETRAINED:
                model = quant_tools.QuantModel(model, config)
                model.allocate_zero_bit(config, logger)
            utils.train.load_ssl_checkpoint(model, path=config.MODEL.PRETRAINED)
            #utils.train.load_ssl_checkpoint(model, path=config.MODEL.PRETRAINED)
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias', 'model.fc.weight', 'model.fc.bias']:
                #print(name, param.size())
                param.requires_grad = False
        #model.fc.weight.data.normal_(mean=0.0, std=0.01)
        #model.fc.bias.data.zero_()


    model = model.to(device)
    
    if not isinstance(model, quant_tools.QuantModel):
        print('quant model')
        model = quant_tools.QuantModel(model, config)
    model._register_hook_update()
    calibration_run(model, train_dataloader, device)
    model._unregister_hook()

    print(model)

    model.allocate_bit(config, logger)

    for m in model.modules():
        #print(n)
        if isinstance(m, quant_tools.QuantLinear):
            print(m)
            if m.output_quantizer:
                m.output_quantizer.bit=0
            if m.weight_quantizer:
                m.weight_quantizer.bit=0
    # I don't want to quantize fc
    #model.fc.output_quantizer.bit = 0
    #model.fc.weight_quantizer.bit = 0
    #if 'Qbyol' not in config.MODEL.PRETRAINED:
    model.set_quant_state(True, True, w_init=True, a_init=True)
    
    print(model)

    if config.TRAIN.USE_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
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
        mixup_fn=mixup_fn,
    )
    eval_runner = runner_class(
        mode="validation",
        model=model,
        config=config,
        dataloader=validation_dataloader,
        logger=logger,
    )

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if config.TRAIN.USE_DDP:
            train_dataloader.sampler.set_epoch(epoch)
        train_runner(model, epoch)
        if utils.ddp.is_main_process():
            eval_runner(model, epoch)
