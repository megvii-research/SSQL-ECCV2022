import torch
import torch.nn as nn


def is_non_negative_act(act_func):
    return isinstance(act_func, (nn.ReLU, nn.ReLU6))
