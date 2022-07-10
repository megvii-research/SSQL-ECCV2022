import abc
import time
import math
import importlib

import torch
from timm.optim import create_optimizer

import meters
import utils
import datasets
from losses import LOSS
from config import update_config
from meters.basic import MeterBasic


class RunnerBase(abc.ABC):
    def __init__(self, mode, model, config, dataloader, logger, **kwargs):
        assert mode in [
            "train",
            "validation",
        ], "mode must be choiced in [train, validation]"
        self.mode = mode
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.logger = logger
        self.mixup_fn = kwargs.get("mixup_fn", None)
        self.data_loader = dataloader
        # build optimizer
        optimizer_args = utils.train.get_optimizer_args(config)
        self.optimizer = create_optimizer(optimizer_args, model)
        
        # build lr-scheduler
        scheduler_args = utils.train.get_scheduler_args(config)
        self.lr_scheduler, total_epochs = utils.create_scheduler(
            scheduler_args, self.optimizer
        )
        if mode == "train":
            update_config(config, "TRAIN.EPOCHS", total_epochs)
            utils.logging_information(logger, config, str(self.optimizer))
            utils.logging_information(logger, config, str(self.lr_scheduler))
        self.meter = importlib.import_module("meters." + config.TRAIN.METER.NAME).METER(
            config
        )
        self.criterion = LOSS(config, model)
        self.batch_time = meters.average.METER()
        self.print_freq = config.TRAIN.PRINT_FREQ
        if self.print_freq > len(self.data_loader):
            self.print_freq = math.ceil(len(self.data_loader) // 2)

    def before_train(self, model, epoch):
        self.model = model
        self.epoch = epoch
        if self.mode == "train":
            model.train()
        else:
            model.eval()
        self.criterion.reset()
        self.batch_time.reset()
        self.meter.reset()
        self.end = time.time()
        self.steps = 0
        if self.mode == "train":
            self.common_fmt = "Epoch: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
        elif self.mode == "validation":
            self.common_fmt = "Test: [{0}][{1}/{2}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t"

    def after_train(self):
        self.lr_scheduler.step(self.epoch)
        self.meter.update_best_meter_dict()
        if self.mode == "validation":
            meter_dict = self.meter.value()
            lr = utils.get_learning_rate(self.config, self.lr_scheduler, self.epoch)
            info_str = " | ".join(
                ["Epoch: {}".format(self.epoch), "lr: {:.5f}".format(lr)]
                + ["{}: {}".format(k, v) for k, v in meter_dict.items()]
            )
            self.logger.info(info_str)
            state_dict = (
                self.model.module.state_dict()
                if self.config.TRAIN.USE_DDP
                else self.model.state_dict()
            )
            utils.train.save_checkpoint(
                {
                    "epoch": self.epoch + 1,
                    "state_dict": state_dict,
                },
                is_best=self.meter.is_best,
                dirname=self.config.OUTPUT,
                filename="checkpoint.pth.tar".format(self.epoch + 1),
            )
            if self.epoch == (self.config.TRAIN.EPOCHS - 1):
                meter_dict = self.meter.get_best_values()
                self.logger.info(
                    "Best Result: "
                    + " | ".join(["{}: {}".format(k, v) for k, v in meter_dict.items()])
                )

    def before_step(self):
        pass

    def after_step(self):
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()
        self.steps += 1
        if self.steps % self.print_freq == 0:
            common_info = self.common_fmt.format(
                self.epoch,
                self.steps,
                len(self.data_loader),
                batch_time=self.batch_time,
            )
            print(common_info + str(self.criterion) + str(self.meter))

    def data_preprocess(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        if self.mixup_fn is not None:
            data, target = self.mixup_fn(data, target)
        return data, target

    @abc.abstractmethod
    def _forward(self):
        pass

    @abc.abstractmethod
    def _backward(self):
        pass

    def __call__(self, model, epoch):
        self.before_train(model, epoch)
        for _, (data, target) in enumerate(self.data_loader):
            data, target = self.data_preprocess(data, target)
            self.before_step()
            output = self._forward(model, data)
            loss = self.criterion(output, target)
            self.meter.update(output, target)
            self._backward(loss)
            self.after_step()
        self.after_train()
        return self.meter.value()
