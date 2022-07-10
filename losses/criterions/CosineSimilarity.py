import torch
from torch import nn
import torch.nn.functional as F
from losses import BaseLoss as Base

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class Criterion(Base):
    def __init__(self, config, model):
        super(Criterion, self).__init__(config, model)
        self.name = "CosineSimilarity"
        self.criterion = nn.CosineSimilarity(dim=1)

    def __call__(self, pred, target):
        if len(pred)==2:
            p1, p2 = pred
            if len(target)==2:
                z1,  z2 = target
                val = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
            else:
                z1, z2, oz1, oz2 = target
                val1 = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
                val2 = -(self.criterion(p1, oz2.detach()).mean() + self.criterion(p2, oz1.detach()).mean()) * 0.5
                val = (val1+val2)*0.5
        else:
            p1, p2, op1, op2 = pred
            if len(target)==2:
                z1, z2 = target
                val1 = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
                val2 = -(self.criterion(op1, z2.detach()).mean() + self.criterion(op2, z1.detach()).mean()) * 0.5
                val = (val1+val2)*0.5
            else:
                z1, z2, oz1, oz2 = target
                val1 = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
                val2 = -(self.criterion(op1, z2.detach()).mean() + self.criterion(op2, z1.detach()).mean()) * 0.5
                val3 = -(self.criterion(op1, oz2.detach()).mean() + self.criterion(op2, oz1.detach()).mean()) * 0.5
                val = (val1+val2+val3)/3

        self.update(val.item(), batch_size=p1.shape[0])
        return val
