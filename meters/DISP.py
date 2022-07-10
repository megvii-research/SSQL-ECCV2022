import numpy as np
import numbers

import torch
import torch.nn.functional as F

from .basic import MeterBasic


def d1_metric(d_est, d_gt, mask, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = (e > 3) & (e / d_gt > 0.05)

    if use_np:
        mean = np.mean(err_mask.astype("float"))
    else:
        mean = torch.mean(err_mask.float())

    return mean


class METER(MeterBasic):
    def __init__(self, config=None, max_disp=192):
        super().__init__()
        self.max_disp = max_disp
        self.best_meter_dict = None
        self.is_best = False
        self.reset()

    def reset(self):
        self.epe_val = 0
        self.epe_avg = 0
        self.epe_sum = 0
        self.d1_val = 0
        self.d1_avg = 0
        self.d1_sum = 0
        self.count = 0
        self.is_best = False

    def update(self, output, target):
        pred_disp = output[-1].squeeze(1)  # [B, H, W]
        gt_disp = target
        mask = (gt_disp > 0) & (gt_disp < self.max_disp)
        if not mask.any():
            return

        if pred_disp.size(-1) < gt_disp.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = (
                F.interpolate(
                    pred_disp,
                    (gt_disp.size(-2), gt_disp.size(-1)),
                    mode="bilinear",
                    align_corners=False,
                )
                * (gt_disp.size(-1) / pred_disp.size(-1))
            )
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction="mean")
        d1 = d1_metric(pred_disp, gt_disp, mask)

        self.count += 1

        self.epe_val = epe.item()
        self.epe_sum += epe.item()
        self.epe_avg = self.epe_sum / self.count
        self.d1_val = d1.item()
        self.d1_sum += d1.item()
        self.d1_avg = self.d1_sum / self.count

    def value(self):
        return {"epe": self.epe_avg, "d1": self.d1_avg}

    def update_best_meter_dict(self):
        cur_meter_dict = self.value()
        if self.best_meter_dict is None:
            self.best_meter_dict = cur_meter_dict
        cur = cur_meter_dict["epe"]
        best = self.best_meter_dict["epe"]
        if cur <= best:
            self.best_meter_dict = cur_meter_dict
            self.is_best = True

    def __repr__(self):
        meter_dict = self.value()
        return "\t".join(["{}: {:.3f}".format(k, v) for k, v in meter_dict.items()])
