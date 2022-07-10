#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import shutil
import random
import builtins

from PIL import ImageFilter
import torch.distributed as dist

from config import update_config


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process():
    return get_rank() == 0


def print_pass(*args):
    pass


def init_distributed_mode(cfg):
    update_config(cfg, "RANK", int(os.environ["RANK"]))
    update_config(cfg, "WORLD_SIZE", int(os.environ["WORLD_SIZE"]))
    update_config(cfg, "LOCAL_RANK", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(cfg.LOCAL_RANK)
    dist_url = "env://"
    print("| distributed init (rank {}): {}".format(cfg.RANK, dist_url), flush=True)
    torch.distributed.init_process_group(
        backend="nccl", init_method=dist_url, world_size=cfg.WORLD_SIZE, rank=cfg.RANK
    )
    torch.distributed.barrier()
    setup_for_distributed(cfg.RANK == 0)
