import os
import random
import importlib
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from timm.optim import create_optimizer
import torchvision.transforms as transforms
from PIL import Image
from config import get_config, update_config
import datasets
import models
import quant_tools
import utils
import runners
import math
import shutil
from quant_tools.blocks import (
    QuantConv2d,
    QuantLinear,
    QuantFeature,
    NAME_QBLOCK_MAPPING,
    QuantBasic,
)
from models import BLOCK_NAME_MAPPING


def calibration_run(model, dataloader, device):
    model.eval()
    tmp_size = len(dataloader)
    #self._register_hook(calibration_init_param)
    iter_dataloader = iter(dataloader)
    cnt = 0
    while tmp_size > 0:
        cnt += 1
        data = next(iter_dataloader)
        if isinstance(data, (list, tuple)):
            data = data[0]
        with torch.no_grad():
            model(data.to(device))
        tmp_size -= data.shape[0]
        #print(data.shape[0])
        #if cnt == 5:
        #    break
    #if self.cfg.QUANT.A.OBSERVER_METHOD.NAME == "MSE":
    #     for m in self.model.modules():
    #        if isinstance(m, tuple(NAME_QBLOCK_MAPPING.values())):
    #            m.output_quantizer.observer.calc_min_max_val()

def ptq(args):
    config = get_config(args)
    device = torch.device(config.DEVICE)

    # ---- setup logger and output ----
    os.makedirs(args.output, exist_ok=True)
    logger = utils.train.construct_logger("FLOAT", config.OUTPUT)

    cudnn.benchmark = True
    utils.train.set_random_seed(config)

    # build dataloaders
    train_preprocess, eval_preprocess, mixup_fn = datasets.build_transforms(
        config, logger
    )

    print(train_preprocess, eval_preprocess)

    '''
    #for cifar dataset
    if 'cifar' in config.TRAIN.DATASET:
        print('cifar transform')
        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0),
                                        ratio=(3.0 / 4.0, 4.0 / 3.0),
                                        interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        eval_preprocess = transforms.Compose([
            transforms.Resize(int(32 * (8 / 7)), interpolation=Image.BICUBIC),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    #'''

    train_dataloader = datasets.build_dataloader("train", config, train_preprocess)
    calibration_dataloader = datasets.build_dataloader("calibration", config, eval_preprocess)
    validation_dataloader = datasets.build_dataloader(
        "validation", config, eval_preprocess
    )

    model = models.__dict__[config.MODEL.ARCH](num_classes=config.MODEL.NUM_CLASSES)
    if config.MODEL.PRETRAINED:
        model = utils.train.load_checkpoint(model, config.MODEL.PRETRAINED)
    #if 'Qbyol' not in config.MODEL.PRETRAINED:
    model = quant_tools.QuantModel(model, config)   
    
    # resume from checkpoint
    if config.MODEL.CHECKPOINT:
        ckpt = torch.load(config.MODEL.CHECKPOINT, map_location="cpu")
        config.defrost()
        state_dict = ckpt['state_dict']
        print(state_dict.keys())
        print('='*20)
        print(model.state_dict().keys())
        print('='*20)
        msg = model.load_state_dict(state_dict, strict=False)
        print('state dict missing', set(msg.missing_keys))
        print("=> loaded checkpoint model '{}'".format(config.MODEL.CHECKPOINT))

    model = model.to(device)
    print(model)
    model.allocate_bit(config)

    #'''
    model.reset_minmax()
    model._register_hook_update()
    if 'imagenet' in config.TRAIN.DATASET:
        calibration_run(model, calibration_dataloader, device)
    else:
        calibration_run(model, train_dataloader, device)
    model._unregister_hook()
    #model.calibration(calibration_dataloader,  config.QUANT.CALIBRATION.SIZE)

    for m in model.modules():
        if isinstance(m, quant_tools.QuantLinear):
            print(m)
            if m.output_quantizer:
                m.output_quantizer.bit=0
            if m.weight_quantizer:
                m.weight_quantizer.bit=0
    #model.calibration(train_dataloader, config.QUANT.CALIBRATION.SIZE)
    model.set_quant_state(w_quant=True, a_quant=True, w_init=True, a_init=True)
    #'''

    utils.logging_information(logger, config, str(model))

    criterion = nn.CrossEntropyLoss().cuda()


    acc1 = validate(validation_dataloader, model, criterion, config)


def validate(val_loader, model, criterion, config):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            #if args.gpu is not None:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
