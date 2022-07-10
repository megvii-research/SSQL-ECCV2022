import torch
import torch.nn as nn

class SimCLR(nn.Module):
    """
    Build a SimCLR model 
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 128)
         """
        super(SimCLR, self).__init__()

        self.config = config
        dim = config.SSL.SETTING.DIM
        mlp = config.SSL.SETTING.MLP
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)

        prev_dim = self.encoder_q.fc.weight.shape[1]

        fc_dim = hidden_dim
        if mlp:  # hack: brute-force replacement
            self.encoder_q.fc = nn.Sequential(nn.Linear(prev_dim, fc_dim),
                                        #nn.BatchNorm1d(fc_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        #nn.Linear(fc_dim, fc_dim),
                                        #nn.BatchNorm1d(fc_dim),
                                        #nn.ReLU(inplace=True), # second layer
                                        nn.Linear(fc_dim,  dim))


    def forward(self, x1, x2):

        f1 = self.encoder_q(x1)  # queries: NxC
        f1 = nn.functional.normalize(f1, dim=1)

        f2 = self.encoder_q(x2)  # queries: NxC
        f2 = nn.functional.normalize(f2, dim=1)

        return f1, f2