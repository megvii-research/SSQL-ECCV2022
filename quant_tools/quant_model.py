import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_tools.blocks import (
    QuantConv2d,
    QuantLinear,
    QuantFeature,
    NAME_QBLOCK_MAPPING,
    QuantBasic,
)
from models import BLOCK_NAME_MAPPING
from . import autobit


class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, config):
        super().__init__()
        self.model = model
        self.cfg = config
        self.device = torch.device(config.DEVICE)
        self.quant_module_refactor(self.model)
        self.build_quantizer(config)

    def forward(self, x):
        out = self.model(x)
        return out 
    
    def quant_module_refactor(self, module: nn.Module):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule
        """
        prev_quantmodule = None
        for n, m in module.named_children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                QuantModule = QuantConv2d if isinstance(m, nn.Conv2d) else QuantLinear
                setattr(module, n, QuantModule(m, self.cfg))
                prev_quantmodule = getattr(module, n)
            elif isinstance(m, tuple(BLOCK_NAME_MAPPING.keys())):
                print(n)
                setattr(
                    module,
                    n,
                    NAME_QBLOCK_MAPPING[BLOCK_NAME_MAPPING[m.__class__]](m, self.cfg),
                )
                prev_quantmodule = getattr(module, n)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if prev_quantmodule is not None:
                    prev_quantmodule.bn_function = m
                    setattr(module, n, nn.Identity())
                else:  # relu前面是个elemenwise-opr
                    continue
            elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Hardtanh, nn.LeakyReLU)):
                if prev_quantmodule is not None:
                    prev_quantmodule.act_function = m
                    setattr(module, n, nn.Identity())
                else:  # relu前面是个elemenwise-opr
                    continue
            elif isinstance(m, nn.Identity):
                continue
            else:
                prev_quantmodule = self.quant_module_refactor(m)
        return prev_quantmodule

    def reset_quant_params(self):
        for m in self.model.modules():
            if isinstance(m, tuple(NAME_QBLOCK_MAPPING.values())):
                m.reset_quant_params()

    def set_quant_state(self, w_quant=False, a_quant=False, w_init=False, a_init=False):
        for m in self.model.modules():
            if isinstance(m, tuple(NAME_QBLOCK_MAPPING.values())):
                m.set_quant_state(w_quant, a_quant, w_init, a_init)

    def build_quantizer(self, config):
        for m in self.model.modules():
            if isinstance(m, QuantBasic) or isinstance(m, QuantFeature):
                m.build_quantizer(config)
            
    def _unregister_hook(self):
        for h in self.forward_hook_handles:
            h.remove()
                #for m in self.model.modules():
        #    if isinstance(m, tuple(NAME_QBLOCK_MAPPING.values())):
        #        m.register_forward_hook(hook=_forward_hook)
    
    def _register_hook_update(self, momentum=None):
        def _forward_hook(module, x_in, x_out):
            if hasattr(module, "output_quantizer") and module.output_quantizer:
                if momentum:
                    module.output_quantizer.observer.update(x_out.detach(), momentum)
                else:
                    module.output_quantizer.observer.update(x_out.detach())
            if hasattr(module, "weight_quantizer") and module.weight_quantizer:
                if momentum:
                    module.weight_quantizer.observer.update(module.weight, momentum)
                else:
                    module.weight_quantizer.observer.update(module.weight)
        self.forward_hook_handles = []
        for m in self.model.modules():
            #print(type(m))
            if isinstance(m, QuantBasic):
                self.forward_hook_handles.append(
                    m.register_forward_hook(hook=_forward_hook)
                )

    def _register_hook(self, calibration_init_param=True):
        def _forward_hook(module, x_in, x_out):
            if hasattr(module, "output_quantizer") and module.output_quantizer:
                if calibration_init_param:
                    module.output_quantizer.observer.calc_init_params(x_out.detach())
                else:
                    module.output_quantizer.observer.update(x_out.detach())

        self.forward_hook_handles = []
        for m in self.model.modules():
            if isinstance(m, tuple(NAME_QBLOCK_MAPPING.values())):
                self.forward_hook_handles.append(
                    m.register_forward_hook(hook=_forward_hook)
                )

    def calibration(self, dataloader, size):
        if size != 0:
            self.model.eval()
            if size < 0:
                size = len(dataloader) * dataloader.batch_size
            calibration_init_param = True
            for i in range(2):
                tmp_size = size
                self._register_hook(calibration_init_param)
                iter_dataloader = iter(dataloader)
                while tmp_size > 0:
                    data = next(iter_dataloader)
                    if isinstance(data, (list, tuple)):
                        data = data[0]
                    with torch.no_grad():
                        self.model(data.to(self.device))
                    tmp_size -= data.shape[0]
                for h in self.forward_hook_handles:
                    h.remove()
                calibration_init_param = False
        #if self.cfg.QUANT.A.OBSERVER_METHOD.NAME == "MSE":
        #    for m in self.model.modules():
        #        if isinstance(m, tuple(NAME_QBLOCK_MAPPING.values())):
        #            m.output_quantizer.observer.calc_min_max_val()

    def allocate_bit(self, cfg, logger=None):
        if cfg.QUANT.BIT_ASSIGNER.NAME:
            bit_assigner = autobit.__dict__[cfg.QUANT.BIT_ASSIGNER.NAME](cfg, logger)
            bit_config = bit_assigner(self)
        else:
            bit_config = autobit.get_uniform_bit_config(self.model, cfg)
        #print(bit_config.keys())
        if len(cfg.QUANT.BIT_CONFIG) > 0:
            specific_bit_config = cfg.QUANT.BIT_CONFIG[0]
            for n, v in specific_bit_config.items():
                if v.get("w", None) is not None:
                    bit_config[n]["w"] = v.get("w")
                if v.get("a", None) is not None:
                    bit_config[n]["a"] = v.get("a")
        self._set_bit(bit_config)

    def allocate_zero_bit(self, cfg, logger=None):
        bit_config = autobit.get_zero_bit_config(self.model, cfg)
        if len(cfg.QUANT.BIT_CONFIG) > 0:
            specific_bit_config = cfg.QUANT.BIT_CONFIG[0]
            for n, v in specific_bit_config.items():
                if v.get("w", None) is not None:
                    bit_config[n]["w"] = 0
                if v.get("a", None) is not None:
                    bit_config[n]["a"] = 0
        self._set_bit(bit_config)

    def _set_bit(self, bit_cfg):
        for n, m in self.model.named_modules():
            if n in bit_cfg.keys():
                if "w" in bit_cfg[n].keys():
                    if hasattr(m, "weight_quantizer") and m.weight_quantizer:
                        m.weight_quantizer.set_bit(bit_cfg[n]["w"])
                    else:
                        raise ValueError("Config[%s][w] mismatch" % n)
                if "a" in bit_cfg[n].keys():
                    if hasattr(m, "output_quantizer") and m.output_quantizer:
                        m.output_quantizer.set_bit(bit_cfg[n]["a"])
                    else:
                        raise ValueError("Config[%s][a] mismatch" % n)
                bit_cfg.pop(n)
        if len(bit_cfg.keys()) > 0:
            raise ValueError("{} mismatch".format(bit_cfg))

    def _print_calibration_info(self):
        for m in self.model.modules():
            if isinstance(m, tuple(NAME_QBLOCK_MAPPING.values())):
                print(m.output_quantizer.observer)
                print(m.output_quantizer)
                break

    def reset_minmax(self):
        for m in self.model.modules():
            #print(n)
            if isinstance(m, QuantBasic):
                if m.output_quantizer:
                    m.output_quantizer.observer.reset_minmax()
                if m.weight_quantizer:
                    m.weight_quantizer.observer.reset_minmax()
           # print('='*10)

    def calibration_new(self, dataloader, size):
        if size != 0:
            self.model.eval()
            if size < 0:
                size = len(dataloader) * dataloader.batch_size
            calibration_init_param = False

            tmp_size = size
            self._register_hook(calibration_init_param)
            iter_dataloader = iter(dataloader)
            while tmp_size > 0:
                data = next(iter_dataloader)
                if isinstance(data, (list, tuple)):
                    data = data[0]
                with torch.no_grad():
                    self.model(data.to(self.device))
                tmp_size -= data.shape[0]
                break
            for h in self.forward_hook_handles:
                h.remove()
