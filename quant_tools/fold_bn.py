import torch
from torch import nn as nn


__all__ = ["search_fold_and_remove_bn"]


def _fold_bn(conv_module, bn_module):
    w = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)
    if bn_module.affine:
        weight = w * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)
    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w
    # set bn running stats for identity transformation
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data ** 2


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return (isinstance(m, nn.Conv2d)) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for n, m in model.named_children():
        if (
            is_bn(m)
            and is_absorbing(prev)
            and m.weight.shape[0] == prev.weight.shape[0]
        ):
            fold_bn_into_conv(prev, m)
            # set the bn module to straight through
            setattr(model, n, nn.Identity())
        elif is_absorbing(m):
            prev = m
        elif is_bn(m):
            # set bn without qweight module to affine
            setattr(model, n, Affine(m))
        else:
            prev = search_fold_and_remove_bn(m)
    return prev


class Affine(nn.Module):
    def __init__(self, bn):
        super(Affine, self).__init__()
        self.affine = bn.affine
        if self.affine:
            self.k = bn.weight.reshape(1, bn.weight.shape[0], 1, 1)
            self.b = bn.bias.reshape(1, bn.bias.shape[0], 1, 1)
        self.running_mean = bn.running_mean.reshape(1, bn.running_mean.shape[0], 1, 1)
        running_var = bn.running_var
        self.safe_std = torch.sqrt(running_var + bn.eps).reshape(
            1, running_var.shape[0], 1, 1
        )

    def forward(self, x_in: torch.Tensor):
        x_normalized = (x_in - self.running_mean) / self.safe_std
        if not self.affine:
            return x_normalized
        out = self.k * x_normalized + self.b
        return out
