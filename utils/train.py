import torch
import os
import logging
import datetime
import shlex
import subprocess
import shutil
import random

from timm.scheduler import create_scheduler as timm_create_scheduler
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn

from .ddp import is_main_process, get_rank


class EmptyClass(object):
    pass


def set_random_seed(config):
    if config.SEED is not None:
        random.seed(config.SEED)
        seed = config.SEED + get_rank()
        torch.manual_seed(seed)
        cudnn.deterministic = True


def get_optimizer_args(cfg):
    args = EmptyClass()
    args.opt = cfg.TRAIN.OPTIMIZER.NAME
    args.lr = cfg.TRAIN.LR_SCHEDULER.BASE_LR
    args.weight_decay = cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY
    args.momentum = cfg.TRAIN.OPTIMIZER.MOMENTUM
    args.opt_eps = cfg.TRAIN.OPTIMIZER.EPS
    args.opt_betas = cfg.TRAIN.OPTIMIZER.BETAS
    return args


def get_scheduler_args(cfg):
    args = EmptyClass()
    args.epochs = cfg.TRAIN.EPOCHS
    args.sched = cfg.TRAIN.LR_SCHEDULER.TYPE
    args.min_lr = cfg.TRAIN.LR_SCHEDULER.MIN_LR
    args.warmup_lr = cfg.TRAIN.LR_SCHEDULER.WARMUP_LR
    args.warmup_epochs = cfg.TRAIN.LR_SCHEDULER.WARMUP_EPOCHS
    args.specific_lrs = cfg.TRAIN.LR_SCHEDULER.SPECIFIC_LRS
    args.decay_rate = cfg.TRAIN.LR_SCHEDULER.DECAY_RATE
    args.decay_epoch = cfg.TRAIN.LR_SCHEDULER.DECAY_EPOCH
    args.decay_milestones = cfg.TRAIN.LR_SCHEDULER.DECAY_MILESTONES
    args.cooldown_epochs = 0  # ??
    return args


def create_scheduler(args, optimizer):
    if args.sched == "multiStep":
        if args.warmup_epochs > 0:
            raise NotImplementedError("TODO warmup in MultiStepLR")
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=args.decay_milestones,
            gamma=args.decay_rate,
        )
        num_epochs = args.epochs
    else:
        lr_scheduler, num_epochs = timm_create_scheduler(args, optimizer)

    return lr_scheduler, num_epochs


def construct_logger(name, save_dir):
    def git_hash():
        cmd = 'git log -n 1 --pretty="%h"'
        ret = subprocess.check_output(shlex.split(cmd)).strip()
        if isinstance(ret, bytes):
            ret = ret.decode()
        return ret

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    date = str(datetime.datetime.now().strftime("%m%d%H%M"))
    fh = logging.FileHandler(os.path.join(save_dir, f"log-{date}.txt"))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def save_checkpoint(state, is_best, dirname, filename):
    if is_main_process():
        dst_pathname = os.path.join(dirname, filename)
        torch.save(state, dst_pathname)
        if is_best:
            shutil.copyfile(dst_pathname, os.path.join(dirname, "model_best.pth.tar"))


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt)
    return model

def load_checkpoint_wo_fc(model, path):
    ckpt = torch.load(path, map_location="cpu")
    print(model.state_dict().keys())
    print(ckpt.keys())
    print('='*10)
    for k in list(ckpt.keys()):
        if 'fc' in k:
            del ckpt[k]
    msg  = model.load_state_dict(ckpt, strict=False)
    print(msg.missing_keys)
    return model


def load_from_Qmodel(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt['state_dict']
    print(state_dict.keys())
    print('='*20)
    #print(model.state_dict().keys())
    model_dict = model.state_dict().keys()
    print(model_dict)
    print('='*20)
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder_q.model') and not k.startswith('encoder_q.model.fc'):
            # remove prefix
            if k[len("encoder_q.model."):] in model_dict:
                print(k[len("encoder_q.model."):])
                state_dict[k[len("encoder_q.model."):]] = state_dict[k]
        del state_dict[k]
    
    msg = model.load_state_dict(state_dict, strict=False)
    #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print('missing', set(msg.missing_keys))
    print("=> loaded pre-trained model '{}'".format(path))
    return model
        
def load_ssl_checkpoint(model, path, warmup_fc=False):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt['state_dict']
    print(state_dict.keys())
    print('='*20)
    print(model.state_dict().keys())
    print('='*20)

    if not warmup_fc:
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            #if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') and 'quantizer' not in k:
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') and 'quantizer' not in k:
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            elif k.startswith('module.encoder') and not k.startswith('module.encoder.fc') and 'quantizer' not in k:
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            elif k.startswith('encoder_q') and not k.startswith('encoder_q.fc') and 'quantizer' not in k:
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            elif k.startswith('module') and not k.startswith('module.fc') and 'quantizer' not in k:
                state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
    
    msg = model.load_state_dict(state_dict, strict=False)
    #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print('missing', set(msg.missing_keys))
    print("=> loaded pre-trained model '{}'".format(path))
    return model

def logging_information(logger, cfg, info):
    if is_main_process():
        logger.info(info)


def get_learning_rate(config, scheduler, epoch):
    if config.TRAIN.LR_SCHEDULER.TYPE in ["multiStep", "specificStep"]:
        lr = scheduler.get_last_lr()[0]
    else:
        lr = scheduler._get_lr(epoch)[0]
    return lr


class LinearTempDecay:
    def __init__(
        self,
        t_max: int,
        rel_start_decay: float = 0.2,
        start_b: int = 10,
        end_b: int = 2,
    ):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
