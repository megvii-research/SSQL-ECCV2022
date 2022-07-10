import importlib


def build_quantizer(config, c_axis, **kwargs):
    prefix = "quant_tools.quantizers.qat."
    module = importlib.import_module(prefix + config.QUANTIZER)
    if c_axis == 0:
        quantizer = module.QWeight(config, kwargs["weight"])
    elif c_axis == 1:
        quantizer = module.QAct(config, kwargs["signed"])
    return quantizer
