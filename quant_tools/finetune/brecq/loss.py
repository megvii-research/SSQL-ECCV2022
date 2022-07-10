import torch
from quant_tools.blocks import (
    QuantConv2d,
    QuantLinear,
    QuantBasic,
    NAME_QBLOCK_MAPPING,
)
from utils import LinearTempDecay
from losses.criterions.LpLoss import Criterion as LPLOSS


class LossFunction:
    def __init__(
        self,
        block: QuantBasic,
        round_loss: str = "relaxation",
        weight: float = 1.0,
        rec_loss: str = "mse",
        max_count: int = 2000,
        b_range: tuple = (10, 2),
        decay_start: float = 0.0,
        warmup: float = 0.0,
        p: float = 2.0,
    ):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.lp_loss = LPLOSS(p=self.p)
        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1],
        )
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == "mse":
            rec_loss = self.lp_loss(pred, tgt)
        elif self.rec_loss == "fisher_diag":
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == "fisher_full":
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError(
                "Not supported reconstruction loss function: {}".format(self.rec_loss)
            )

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == "none":
            b = round_loss = 0
        elif self.round_loss == "relaxation":
            round_loss = 0
            if isinstance(self.block, (QuantConv2d, QuantLinear)):
                round_vals = self.block.weight_quantizer.get_soft_targets()
                round_loss += (
                    self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
                )
            else:
                for name, module in self.block.named_modules():
                    if isinstance(module, (QuantConv2d, QuantLinear)):
                        round_vals = module.weight_quantizer.get_soft_targets()
                        round_loss += (
                            self.weight
                            * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
                        )
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            print(
                "Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}".format(
                    float(total_loss), float(rec_loss), float(round_loss), b, self.count
                )
            )
        return total_loss
