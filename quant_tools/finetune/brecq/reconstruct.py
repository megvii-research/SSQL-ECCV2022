import torch
import torch.nn as nn

from quant_tools import QuantModel
from quant_tools.blocks import QuantBasic, QuantBasicBlock, QuantConv2d, QuantLinear

from .loss import LossFunction
from .train import train


def reconstruct_modules(
    model: QuantModel,
    module: QuantBasic,
    cali_data: torch.Tensor,
    batch_size: int = 32,
    iters: int = 20000,
    weight: float = None,
    opt_mode: str = "mse",
    asym: bool = False,
    b_range: tuple = (20, 2),
    warmup: float = 0.0,
    act_quant: bool = False,
    lr: float = 4e-5,
    p: float = 2.0,
    multi_gpu: bool = False,
    keep_gpu: bool = True,
):
    """
    Module reconstruction to optimize the output from each module.

    :param model: QuantModel
    :param module: QuantBasic that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    """
    model.set_quant_state(False, False)
    module.set_quant_state(True, act_quant)
    round_mode = "learned_hard_sigmoid"
    weight = 1e-2 if isinstance(module, QuantBasicBlock) else 1e-3

    opt_params = []
    if not act_quant:
        for n, m in module.named_modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                m.weight_quantizer.soft_targets = True
                opt_params += [m.weight_quantizer.alpha]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        for n, m in module.named_modules():
            if isinstance(m, QuantBasic) and m.output_quantizer.scale is not None:
                m.output_quantizer.scale = nn.Parameter(
                    m.output_quantizer.scale.detach()
                )
                m.output_quantizer.zero_point = m.output_quantizer.zero_point.detach()
                opt_params += [m.output_quantizer.scale]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=iters, eta_min=0.0
        )

    loss_mode = "none" if act_quant else "relaxation"
    rec_loss = opt_mode

    loss_func = LossFunction(
        module,
        round_loss=loss_mode,
        weight=weight,
        max_count=iters,
        rec_loss=rec_loss,
        b_range=b_range,
        decay_start=0,
        warmup=warmup,
        p=p,
    )

    train(
        model,
        module,
        cali_data,
        loss_func,
        optimizer,
        scheduler,
        batch_size,
        opt_mode,
        iters,
        act_quant,
        asym,
        keep_gpu=keep_gpu,
    )

    # Finish optimization, use hard rounding.
    for n, m in module.named_modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            m.weight_quantizer.soft_targets = False
