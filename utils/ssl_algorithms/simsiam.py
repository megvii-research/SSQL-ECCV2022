import torch
import torch.nn as nn
from copy import deepcopy
import quant_tools

class SimSiam(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        self.config = config
        
        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_q = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder_q.fc.weight.shape[1]

        # to make resnet-18 and resnet-34 have larger hidden dimensions, e.g., 2048
        fc_dim = hidden_dim
        self.encoder_q.fc = nn.Sequential(nn.Linear(prev_dim, fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fc_dim, fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim,  dim),
                                        nn.BatchNorm1d(dim, affine=False)) # output layer


        self.encoder_q.fc[6].bias.requires_grad = False #hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        z1 = self.encoder_q(x1)
        z2 = self.encoder_q(x2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return [p1, p2] , [z1.detach(), z2.detach()]
