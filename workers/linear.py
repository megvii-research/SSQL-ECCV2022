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

def linear(args):
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
    train_preprocess, eval_preprocess, mixup_fn = datasets.build_transforms(config, logger)

    #'''
    #for cifar dataset
    if 'cifar' in config.TRAIN.DATASET and 'cifar' in config.MODEL.ARCH:
        print('cifar transform')
        input_shape = config.MODEL.INPUTSHAPE[0]
        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(input_shape, scale=(0.8, 1.0),
                                        ratio=(3.0 / 4.0, 4.0 / 3.0),
                                        interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        eval_preprocess = transforms.Compose([
            transforms.Resize(int(input_shape * (8 / 7)), interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_shape),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    #'''
    if 'imagenet' in config.TRAIN.DATASET:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        print('imagenet transform')
        train_preprocess = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        eval_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    print(train_preprocess, eval_preprocess)

    #'''

    train_dataloader = datasets.build_dataloader("train", config, train_preprocess)
    validation_dataloader = datasets.build_dataloader("validation", config, eval_preprocess)

    model = models.__dict__[config.MODEL.ARCH](num_classes=config.MODEL.NUM_CLASSES)
    quant_flag = False
    if config.MODEL.PRETRAINED:
        if 'Q' in config.MODEL.PRETRAINED or config.TRAIN.WARMUP_FC:
            model = quant_tools.QuantModel(model, config)
            quant_flag = True
        # load from a linear evaluation model or ImageNet model
        #if config.TRAIN.WARMUP_FC:
        #    utils.train.load_checkpoint(model, path=config.MODEL.PRETRAINED)
        # load from a ssl pretained model
        #else:
        # for ImageNet model
        print(quant_flag)
        if 'torch' in config.MODEL.PRETRAINED:
            utils.train.load_checkpoint_wo_fc(model, path=config.MODEL.PRETRAINED)
        else:
            utils.train.load_ssl_checkpoint(model, path=config.MODEL.PRETRAINED, warmup_fc=config.TRAIN.WARMUP_FC)

    if config.TRAIN.LINEAR_EVAL:
        print('fix backebone parameters')
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias', 'model.fc.weight', 'model.fc.bias']:
                #print(name, param.size())
                param.requires_grad = False

    if not quant_flag:
        model = quant_tools.QuantModel(model, config)   
    
    # resume from checkpoint
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
        print("=> loaded checkpoint model '{}'".format(config.MODEL.CHECKPOINT))

    model = model.to(device)
    print(model)
    model.allocate_bit(config)

    # calibration first
    model.reset_minmax()
    model._register_hook_update()
    calibration_run(model, train_dataloader, device)
    model._unregister_hook()

    # we don't quantized the final linear layer
    for m in model.modules():
        if isinstance(m, quant_tools.QuantLinear):
            if m.output_quantizer:
                m.output_quantizer.bit=0
            if m.weight_quantizer:
                m.weight_quantizer.bit=0
    model.set_quant_state(w_quant=True, a_quant=True, w_init=True, a_init=True)

    if config.TRAIN.USE_DDP:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config.LOCAL_RANK],
            find_unused_parameters=True,
        )
    utils.logging_information(logger, config, str(model))

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), config.TRAIN.LR_SCHEDULER.BASE_LR,
                                momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)

    if config.TRAIN.OPTIMIZER.NAME == 'lars':
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    #load optimizer
    if config.MODEL.CHECKPOINT:
        ckpt = torch.load(config.MODEL.CHECKPOINT, map_location="cpu")
        optimizer.load_state_dict(ckpt['optimizer'])

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if config.TRAIN.USE_DDP:
            train_dataloader.sampler.set_epoch(epoch)
        
        adjust_learning_rate(optimizer, config.TRAIN.LR_SCHEDULER.BASE_LR, epoch, config)

        train(train_dataloader, model, criterion, optimizer, epoch, config)

        if epoch>0 and epoch % 20==0 or epoch==config.TRAIN.EPOCHS-1:
            acc1 = validate(validation_dataloader, model, criterion, config)

        state_dict = (
            model.module.state_dict()
            if config.TRAIN.USE_DDP
            else model.state_dict()
        )
        utils.train.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": state_dict,
                'optimizer' : optimizer.state_dict(),
            },
            is_best=False,
            dirname=config.OUTPUT,
            filename="checkpoint.pth.tar".format(epoch+1),
        )


def train(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if config.TRAIN.LINEAR_EVAL:
        model.eval()
    else:
        model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)


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

            #torch.cuda.synchronize()

            if i % 10 == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, init_lr, epoch, config):
    """Decay the learning rate based on schedule"""
    if config.TRAIN.LR_SCHEDULER.TYPE=='cosine':
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / config.TRAIN.EPOCHS))
    else:
        cur_lr = init_lr
        for milestone in config.TRAIN.LR_SCHEDULER.DECAY_MILESTONES:
            cur_lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


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
