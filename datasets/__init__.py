import importlib
from PIL import Image
import torch
from torchvision import transforms

from timm.data.random_erasing import RandomErasing
from timm.data.mixup import Mixup

from .utils import workaround_torch_size_bug
from utils import logging_information
from utils import ddp as ddp_utils
from config import update_config

_INTERPOLATION_MODE_MAPPING = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "box": Image.BOX,
    "hamming": Image.HAMMING,
    "lanczos": Image.LANCZOS,
}


def build_transforms(cfg, logger):
    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)
    mixup_fn = None
    if cfg.AUG.TRAIN.MIX.PROB > 0:
        _cfg = cfg.AUG.TRAIN.MIX
        mixup_fn = Mixup(
            mixup_alpha=_cfg.MIXUP_ALPHA,
            cutmix_alpha=_cfg.CUTMIX_ALPHA,
            cutmix_minmax=_cfg.CUTMIX_MINMAX,
            prob=_cfg.PROB,
            switch_prob=_cfg.SWITCH_MIXUP_CUTMIX_PROB,
            mode=_cfg.MODE,
            label_smoothing=cfg.TRAIN.LABEL_SMOOTHING,
            num_classes=cfg.MODEL.NUM_CLASSES,
        )
    logging_information(logger, cfg, str("train aug: \n") + str(train_transform))
    logging_information(logger, cfg, str("eval aug: \n") + str(eval_transform))
    return train_transform, eval_transform, mixup_fn


def build_train_transform(cfg):
    transform_list = []
    need_resized_input = True  # a flag specified add a resize transform in ending

    #specified for SSL two crop transformations
    #if cfg.SSL.TYPE is not None:
    #    return build_ssl_transform(cfg)
        
    if cfg.AUG.TRAIN.RANDOMRESIZEDCROP.ENABLE:
        _cfg = cfg.AUG.TRAIN.RANDOMRESIZEDCROP
        transform_list.append(
            transforms.RandomResizedCrop(
                size=workaround_torch_size_bug(cfg.MODEL.INPUTSHAPE),
                scale=_cfg.SCALE,
                interpolation=_INTERPOLATION_MODE_MAPPING[_cfg.INTERPOLATION],
            )
        )
        need_resized_input = False

    if cfg.AUG.TRAIN.RANDOMCROP.ENABLE:
        _cfg = cfg.AUG.TRAIN.RANDOMCROP
        transform_list.append(
            transforms.RandomCrop(
                size=workaround_torch_size_bug(cfg.MODEL.INPUTSHAPE),
                padding=_cfg.PADDING,
            )
        )
        need_resized_input = False

    if cfg.AUG.TRAIN.RESIZE.ENABLE:
        _cfg = cfg.AUG.TRAIN.RESIZE
        transform_list.append(
            transforms.Resize(
                size=workaround_torch_size_bug(_cfg.SIZE)
                if _cfg.KEEP_RATIO
                else _cfg.SIZE,
                interpolation=_INTERPOLATION_MODE_MAPPING[_cfg.INTERPOLATION],
            )
        )
        need_resized_input = False

    if cfg.AUG.TRAIN.CENTERCROP.ENABLE:
        transform_list.append(
            transforms.CenterCrop(workaround_torch_size_bug(cfg.MODEL.INPUTSHAPE))
        )
        need_resized_input = False

    if cfg.AUG.TRAIN.HORIZONTAL_FLIP.PROB > 0:
        transform_list.append(
            transforms.RandomHorizontalFlip(p=cfg.AUG.TRAIN.HORIZONTAL_FLIP.PROB)
        )

    if cfg.AUG.TRAIN.VERTICAL_FLIP.PROB > 0:
        transform_list.append(
            transforms.RandomVerticalFlip(p=cfg.AUG.TRAIN.VERTICAL_FLIP.PROB)
        )

    if cfg.AUG.TRAIN.COLOR_JITTER.PROB > 0:
        _cfg = cfg.AUG.TRAIN.COLOR_JITTER
        transform_list.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=_cfg.BRIGHTNESS,
                        contrast=_cfg.CONTRAST,
                        saturation=_cfg.SATURATION,
                        hue=_cfg.HUE,
                    )
                ],
                p=_cfg.PROB,
            )
        )
    if need_resized_input:
        transform_list.append(
            transforms.Resize(workaround_torch_size_bug(cfg.MODEL.INPUTSHAPE))
        )

    transform_list.append(transforms.ToTensor())

    if cfg.AUG.TRAIN.NORMLIZATION.MEAN:
        transform_list.append(
            transforms.Normalize(mean=cfg.AUG.TRAIN.NORMLIZATION.MEAN, std=[1, 1, 1])
        )
    if cfg.AUG.TRAIN.NORMLIZATION.STD:
        transform_list.append(
            transforms.Normalize(
                mean=[0, 0, 0],
                std=cfg.AUG.TRAIN.NORMLIZATION.STD,
            )
        )

    if cfg.AUG.TRAIN.RANDOMERASE.PROB > 0:
        _cfg = cfg.AUG.TRAIN.RANDOMERASE
        transform_list.append(
            RandomErasing(_cfg.PROB, mode=_cfg.MODE, max_count=_cfg.MAX_COUNT)
        )

    return transforms.Compose(transform_list)


def build_ssl_transform(cfg, two_crop=True):
    from datasets.custom_transforms import GaussianBlur, TwoCropsTransform
    normalize =  transforms.Normalize(mean=cfg.AUG.TRAIN.NORMLIZATION.MEAN, 
                                std=cfg.AUG.TRAIN.NORMLIZATION.STD)
    if 'cifar' in cfg.TRAIN.DATASET:
        print('cifar ssl transform')
        transform_list = [transforms.RandomResizedCrop(cfg.MODEL.INPUTSHAPE, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5), # remove gaussianblur for cifar
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize]
    else:
        print('imagenet ssl transform')
        transform_list = [transforms.RandomResizedCrop(cfg.MODEL.INPUTSHAPE, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize]
    if two_crop:
        return TwoCropsTransform(transforms.Compose(transform_list))
    else:
        return transforms.Compose(transform_list)

def build_eval_transform(cfg):
    transform_list = []
    need_resized_input = True  # a flag specified add a resize transform in ending

    if cfg.AUG.EVALUATION.RESIZE.ENABLE:
        _cfg = cfg.AUG.EVALUATION.RESIZE
        transform_list.append(
            transforms.Resize(
                size=workaround_torch_size_bug(_cfg.SIZE)
                if _cfg.KEEP_RATIO
                else _cfg.SIZE,
                interpolation=_INTERPOLATION_MODE_MAPPING[_cfg.INTERPOLATION],
            )
        )
        need_resized_input = False

    if cfg.AUG.EVALUATION.CENTERCROP.ENABLE:
        transform_list.append(
            transforms.CenterCrop(workaround_torch_size_bug(cfg.MODEL.INPUTSHAPE))
        )
        need_resized_input = False

    if need_resized_input:
        transform_list.append(
            transforms.Resize(workaround_torch_size_bug(cfg.MODEL.INPUTSHAPE))
        )

    transform_list.append(transforms.ToTensor())

    if cfg.AUG.EVALUATION.NORMLIZATION.MEAN:
        transform_list.append(
            transforms.Normalize(
                mean=cfg.AUG.EVALUATION.NORMLIZATION.MEAN, std=[1, 1, 1]
            )
        )
    if cfg.AUG.EVALUATION.NORMLIZATION.STD:
        transform_list.append(
            transforms.Normalize(
                mean=[0, 0, 0],
                std=cfg.AUG.EVALUATION.NORMLIZATION.STD,
            )
        )

    return transforms.Compose(transform_list)


def build_dataloader(mode, config, preprocess):
    if mode == "calibration":
        if config.QUANT.CALIBRATION.TYPE == "tar":
            dataset = importlib.import_module("datasets.tardata").DATASET(
                config.QUANT.CALIBRATION.PATH, preprocess
            )
            if config.QUANT.CALIBRATION.SIZE > len(dataset):
                update_config(config, "QUANT.CALIBRATION.SIZE", len(dataset))
        elif config.QUANT.CALIBRATION.TYPE == "python_module":
            raise NotImplementedError("TODO")
        else:
            raise NotImplementedError(
                "No support {}".format(config.QUANT.CALIBRATION.TYPE)
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.QUANT.CALIBRATION.BATCHSIZE,
            shuffle=False,
            num_workers=config.QUANT.CALIBRATION.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        dataset_class = importlib.import_module("datasets." + config.TRAIN.DATASET).DATASET
        dataset = dataset_class(transform=preprocess, mode=mode)
        #for moco we need to drop last
        drop = True if config.SSL.TYPE and 'MoCo' in config.SSL.TYPE else False
        if mode == "train":
            if config.TRAIN.USE_DDP:
                num_tasks = ddp_utils.get_world_size()
                global_rank = ddp_utils.get_rank()
                train_sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                train_sampler = None
            dataloader = torch.utils.data.DataLoader(
                dataset,
                sampler=train_sampler,
                batch_size=config.TRAIN.BATCH_SIZE,  # per-gpu
                num_workers=config.TRAIN.NUM_WORKERS,
                pin_memory=True,
                drop_last = drop,
                shuffle=True if train_sampler is None else False,
            )
        elif mode == "validation":
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=min(256, 2*config.TRAIN.BATCH_SIZE),
                num_workers=config.TRAIN.NUM_WORKERS,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )
    return dataloader
