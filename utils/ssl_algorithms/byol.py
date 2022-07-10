import torch
import torch.nn as nn
from copy import deepcopy
import quant_tools

class BYOL(nn.Module):
    """
    Build a BYOL model.
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(BYOL, self).__init__()

        self.config = config
        
        dim = config.SSL.SETTING.DIM
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM
        self.m = config.SSL.SETTING.MOMENTUM

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder_q = base_encoder(num_classes=dim)

        # build a 3-layer projector
        prev_dim = self.encoder_q.fc.weight.shape[1]
        fc_dim = hidden_dim

        self.encoder_q.fc = nn.Sequential(nn.Linear(prev_dim,fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(fc_dim, fc_dim, bias=False),
                                        nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim, dim)) # output layer
        
        self.encoder_k  = deepcopy(self.encoder_q)

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, hidden_dim, bias=False),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(hidden_dim, dim)) # output layer

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            #print(param_q.size(), param_k.size())
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        online_proj_one = self.encoder_q(x1)
        online_proj_two = self.encoder_q(x2)

        online_pred_one = self.predictor(online_proj_one)
        online_pred_two = self.predictor(online_proj_two)

        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            target_proj_one = self.encoder_k(x1)
            target_proj_two = self.encoder_k(x2)
        
        return [online_pred_one, online_pred_two], [target_proj_one.detach(), target_proj_two.detach()]