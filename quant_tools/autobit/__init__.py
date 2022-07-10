from .hawq import *


def get_uniform_bit_config(model, config):
    bit_config = {}
    for n, m in model.named_modules():
        if hasattr(m, "weight_quantizer") and m.weight_quantizer:
            if bit_config.get(n, None) is None:
                bit_config[n] = {}
            bit_config[n]["w"] = config.QUANT.W.BIT  # 7: is a hack for missing model.
        if hasattr(m, "output_quantizer") and m.output_quantizer:
            if bit_config.get(n, None) is None:
                bit_config[n] = {}
            bit_config[n]["a"] = config.QUANT.A.BIT
    return bit_config

def get_zero_bit_config(model, config):
    bit_config = {}
    for n, m in model.named_modules():
        if hasattr(m, "weight_quantizer") and m.weight_quantizer:
            if bit_config.get(n, None) is None:
                bit_config[n] = {}
            bit_config[n]["w"] = 0  # 7: is a hack for missing model.
        if hasattr(m, "output_quantizer") and m.output_quantizer:
            if bit_config.get(n, None) is None:
                bit_config[n] = {}
            bit_config[n]["a"] = 0
    return bit_config