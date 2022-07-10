import time
import torch
import torch.nn as nn
import torch.optim
import meters
import utils
from .basic import RunnerBase
import math 
import random
import numpy as np
import shutil
import os

def adjust_learning_rate(optimizer, init_lr, epoch, config):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / config.TRAIN.EPOCHS))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

class RUNNER(RunnerBase):
    def __init__(self, mode, model, config, dataloader, logger, **kwargs):
        super().__init__(
            mode,
            model,
            config,
            dataloader=dataloader,
            logger=logger,
            **kwargs,
        )
        print(self.optimizer)
        for param_group in self.optimizer.param_groups:
            print(len(param_group['params']), param_group['weight_decay'], param_group['lr'])

        self.optimizer = torch.optim.SGD(model.parameters(), config.TRAIN.LR_SCHEDULER.BASE_LR,
                                momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                                weight_decay=config.TRAIN.OPTIMIZER.WEIGHT_DECAY)
        
        if config.MODEL.CHECKPOINT:
            ckpt = torch.load(config.MODEL.CHECKPOINT, map_location="cpu")
            if 'optimizer' in ckpt.keys():
                self.optimizer.load_state_dict(ckpt['optimizer'])
                #print('optimizer missing', set(msg))
                print("=> loaded pre-trained optimizer")

        #for param_group in self.optimizer.param_groups:
        #    print(len(param_group['params']), param_group['weight_decay'], param_group['lr'])

    def _forward(self, model, im_q, im_k):
        return model(im_q, im_k)

    def _backward(self, loss):
        if self.mode == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def data_preprocess(self, data):
        data = data.to(self.device)
        return data

    def after_train(self):
        state_dict = (
            self.model.module.state_dict()
            if self.config.TRAIN.USE_DDP
            else self.model.state_dict()
        )
        utils.train.save_checkpoint(
                {
                    "epoch": self.epoch + 1,
                    "state_dict": state_dict,
                    'optimizer' : self.optimizer.state_dict(),
                },
                is_best=False,
                dirname=self.config.OUTPUT,
                filename='checkpoint.pth.tar',
            )
        # save for further evaluation
        if self.epoch == (self.config.TRAIN.EPOCHS - 1) or self.epoch % 20 == 0 or self.epoch % 50 == 0:
            if utils.ddp.is_main_process():
                shutil.copyfile(os.path.join(self.config.OUTPUT, 'checkpoint.pth.tar'), os.path.join(self.config.OUTPUT, 'checkpoint_{:04d}.pth.tar'.format(self.epoch)))
    
    def after_step(self):
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()
        self.steps += 1
        #if self.steps % 1 == 0:
        if self.steps % self.print_freq == 0:
            common_info = self.common_fmt.format(
                self.epoch,
                self.steps,
                len(self.data_loader),
                batch_time=self.batch_time,
            )
            print(common_info + str(self.criterion) + str(self.meter))
            #self.logger.info(common_info + str(self.criterion) + str(self.meter))
            utils.logging_information(self.logger, self.config, common_info + str(self.criterion))


    def __call__(self, model, epoch):
        self.before_train(model, epoch)
        adjust_learning_rate(self.optimizer, self.config.TRAIN.LR_SCHEDULER.BASE_LR, epoch, self.config)
        
        '''
        #warmup for momentum minmax
        warmup_epochs = 0
        if epoch<warmup_epochs:
            model.module.momentum = 0
        else:
            #model.module.momentum = 0.8
            model.module.momentum = 1.0*float(epoch-warmup_epochs) / (self.config.TRAIN.EPOCHS-warmup_epochs)
            #model.module.momentum = 1.0 - 0.5 * (1. + math.cos(math.pi * (epoch-warmup_epochs) / (self.config.TRAIN.EPOCHS-warmup_epochs) ))
        '''

        '''
        #if you want to change bit per epoch

        # increase or decrease
        #random_set = list(np.arange(4,17)) # 4,5,...,16
        #random_set = list(np.arange(4,17))[::-1] # 16,15,...,4
        #per_bit_epochs = 1+self.config.TRAIN.EPOCHS // len(random_set)
        #random_w_bit = random_set[(epoch//per_bit_epochs)%len(random_set)]
        #random_a_bit = random_w_bit

        # random select
        #random_w_bit = random.choice(np.arange(4,17))
        #random_a_bit = random.choice(np.arange(4,17))

        # zigzag selection
        random_set = list(np.arange(4,17))
        per_bit_epoch = 1
        cur_val = epoch // per_bit_epoch #e.g., 150//10=15, 15%13 = 2 (cur_index), 15//13 = 1 (reverse set)

        cur_index = cur_val % len(random_set)
        cur_round = cur_val // len(random_set)
        if cur_round % 2 == 1:
            random_set = random_set[::-1]

        random_w_bit = random_set[cur_index]
        random_a_bit = random_w_bit

        # set bit
        model.module.config.defrost()
        model.module.config.QUANT.W.BIT = int(random_w_bit)
        model.module.config.QUANT.A.BIT = int(random_a_bit)
        print(epoch, model.module.config.QUANT.W.BIT, model.module.config.QUANT.A.BIT)
        #'''

        for i, (data, target) in enumerate(self.data_loader):
            im_q, im_k = self.data_preprocess(data[0]), self.data_preprocess(data[1])
            self.before_step()
            output, target = self._forward(model, im_q, im_k)
            #print(torch.CosineSimilarity(quantization_error, contrastive_error, dim=-1))
            #output, target, [quantization_error, contrastive_error] = self._forward(model, im_q, im_k)
            loss = self.criterion(output, target)
            self._backward(loss)
            self.after_step()
            #break
        self.after_train()
        return self.meter.value()
