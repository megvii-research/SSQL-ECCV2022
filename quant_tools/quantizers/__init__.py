import importlib
import torch
from .basic_modules import Quantizer as BaseQuantizer


def build_quantizer(config, c_axis, weight=None, act_func=None):
    prefix = "quant_tools.quantizers.{}.".format(config.QUANT.TYPE)
    if weight is not None:
        assert isinstance(weight, torch.Tensor), "weight must be a Tensor"
        _cfg = config.QUANT.W
        module = importlib.import_module(prefix + _cfg.QUANTIZER).QWeight
        quantizer = module(_cfg, c_axis, weight)
    elif act_func is not None:
        _cfg = config.QUANT.A
        module = importlib.import_module(prefix + _cfg.QUANTIZER).QAct
        quantizer = module(_cfg, c_axis, act_func)
    else:
        raise NotImplementedError("Only support weight_quantizer or act_quantizer")
    return quantizer
