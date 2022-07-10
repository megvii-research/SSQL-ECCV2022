import torch
import numpy as np
import numbers
from collections import OrderedDict

from .basic import MeterBasic


class METER(MeterBasic):
    def __init__(self, config):
        super().__init__()
        self.topk = np.sort(config.TRAIN.METER.ACC.TOPK)
        self.best_meter_dict = None
        self.is_best = False
        self.reset()

    def reset(self):
        self.sum = {v: 0 for v in self.topk}
        self.n = 0
        self.is_best = False

    def update(self, output, target):
        if torch.is_tensor(output):
            output = output.detach().cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = np.atleast_1d(target.detach().cpu().squeeze().numpy())
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        if np.ndim(output) == 1:
            output = output[np.newaxis]
        else:
            assert np.ndim(output) == 2, "wrong output size (1D or 2D expected)"
            assert np.ndim(target) == 1, "target and output do not match"
        assert target.shape[0] == output.shape[0], "target and output do not match"
        topk = self.topk
        maxk = int(topk[-1])  # seems like Python3 wants int and not np.int64
        no = output.shape[0]

        pred = torch.from_numpy(output).topk(maxk, 1, True, True)[1].numpy()
        correct = pred == target[:, np.newaxis].repeat(pred.shape[1], 1)

        for k in topk:
            self.sum[k] += correct[:, 0:k].sum()
        self.n += no

    def value(self):
        return OrderedDict(
            [
                ("prec{}".format(k), (float(self.sum[k]) / self.n) * 100)
                for k in self.topk
            ]
        )

    def update_best_meter_dict(self):
        cur_meter_dict = self.value()
        if self.best_meter_dict is None:
            self.best_meter_dict = cur_meter_dict
        cur = cur_meter_dict["prec{}".format(self.topk[0])]
        best = self.best_meter_dict["prec{}".format(self.topk[0])]
        if cur >= best:
            self.best_meter_dict = cur_meter_dict
            self.is_best = True

    def __repr__(self):
        meter_dict = self.value()
        return "\t".join(["{}: {:.3f}".format(k, v) for k, v in meter_dict.items()])
