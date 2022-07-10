import torch
from torch import nn
import torch.nn.functional as F
from losses import BaseLoss as Base

def nt_xent(f1, f2, t=0.5):
    x = torch.cat([f1,f2], dim=0)
    #print(x.size())
    #x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))

def nt_xentv2(f1, f2, t=0.5):
    x = torch.cat([f1,f2], dim=0)
    batch_size = f1.size(0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(x, x.t().contiguous()) / t)

    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(f1 *f2, dim=-1) / t)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    return loss

class Criterion(Base):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, config, model):
        super(Criterion, self).__init__(config, model)
        self.name = "SimCLR"
        self.temperature = config.SSL.SETTING.T
    
    def __call__(self, f1, f2):

        if len(f1)==2:
            f1, f1_q = f1
            f2, f2_q = f2
            loss1 = nt_xentv2(f1, f2, self.temperature)
            loss2 = nt_xentv2(f1_q, f2_q, self.temperature)
            loss = (loss1+loss2)*0.5
        else:
            loss  = nt_xentv2(f1, f2, self.temperature)

        self.update(loss.item(), batch_size=f1.shape[0])
        return loss
