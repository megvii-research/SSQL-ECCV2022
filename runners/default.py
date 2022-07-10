import time
import torch
import torch.nn as nn

import meters
from .basic import RunnerBase


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

    def _forward(self, model, data):
        if self.mode == "train":
            return model(data)
        else:
            with torch.no_grad():
                return model(data)

    def _backward(self, loss):
        if self.mode == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
