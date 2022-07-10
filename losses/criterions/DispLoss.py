import torch
from torch import nn
import torch.nn.functional as F

from losses import BaseLoss as Base


class Criterion(Base):
    def __init__(self, config, model, max_disp=192):
        super(Criterion, self).__init__(config, model)
        self.max_disp = max_disp

    def __call__(self, preds, target):
        pred_disp_pyramid = [pred.squeeze(1) for pred in preds]

        gt_disp = target
        mask = (gt_disp > 0) & (gt_disp < self.max_disp)

        # FIXME
        if not mask.any():
            return 0

        disp_loss = 0
        # Loss weights
        if len(pred_disp_pyramid) == 5:
            pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
        elif len(pred_disp_pyramid) == 4:
            pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0]
        elif len(pred_disp_pyramid) == 3:
            pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
        elif len(pred_disp_pyramid) == 1:
            pyramid_weight = [1.0]  # highest loss only
        else:
            raise NotImplementedError

        assert len(pyramid_weight) == len(pred_disp_pyramid)
        for k in range(len(pred_disp_pyramid)):
            pred_disp = pred_disp_pyramid[k]
            weight = pyramid_weight[k]

            if pred_disp.size(-1) != gt_disp.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = (
                    F.interpolate(
                        pred_disp,
                        size=(gt_disp.size(-2), gt_disp.size(-1)),
                        mode="bilinear",
                        align_corners=False,
                    )
                    * (gt_disp.size(-1) / pred_disp.size(-1))
                )
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]

            curr_loss = F.smooth_l1_loss(
                pred_disp[mask], gt_disp[mask], reduction="mean"
            )
            disp_loss += weight * curr_loss

        batch_size = preds[0].shape[0] if type(preds) is list else preds.shape[0]
        self.update(disp_loss.item(), batch_size=batch_size)

        return disp_loss
