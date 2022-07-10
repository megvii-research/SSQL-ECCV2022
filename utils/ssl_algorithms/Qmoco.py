import torch
import torch.nn as nn
from copy import deepcopy
import quant_tools
import utils
import random
import numpy as np
from .moco import MoCo

# ECCV rebuttal: SSQL+MoCov2
class SSQL_MoCo(MoCo):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, config):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SSQL_MoCo, self).__init__(base_encoder, config)

        self.config = config
        dim = config.SSL.SETTING.DIM
        
        self.K = config.SSL.SETTING.MOCO_K
        self.m =  config.SSL.SETTING.MOMENTUM
        self.T = config.SSL.SETTING.T
        mlp = config.SSL.SETTING.MLP
        hidden_dim = config.SSL.SETTING.HIDDEN_DIM

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)
        print('==================================================')
        for k,v in self.encoder_k.state_dict().items():
            print(k,v.size())

        if mlp:  # hack: brute-force replacement
            #dim_mlp = self.encoder_q.fc.weight.shape[1]
            prev_dim = self.encoder_q.fc.weight.shape[1]
            fc_dim = hidden_dim
            self.encoder_q.fc = nn.Sequential(nn.Linear(prev_dim, fc_dim),
                                                     nn.ReLU(),
                                                        nn.Linear(fc_dim, dim))
            self.encoder_k.fc = nn.Sequential(nn.Linear(prev_dim, fc_dim),
                                                     nn.ReLU(),
                                                        nn.Linear(fc_dim, dim))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        self.encoder_q =  quant_tools.QuantModel(self.encoder_q, self.config)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.w_range = self.config.QUANT.W.BIT_RANGE
        self.a_range = self.config.QUANT.A.BIT_RANGE
        print('weight bit range {}, activation bit range {}'.format(self.w_range, self.a_range))

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # set encoder_q to floating-point
        self.encoder_q.set_quant_state(False, False, w_init=False, a_init=False)
        # each iteration we need to recalibrate because the network has been updated
        self.encoder_q.reset_minmax()
        self.encoder_q._register_hook_update() 
        # use floating-point network forward pass for calibration
        q_f = self.encoder_q(im_q)
        q_f = nn.functional.normalize(q_f, dim=1)
        self.encoder_q._unregister_hook()

        # hack: randomly choose bit in each iteration
        #random_w_bit = random.choice(np.arange(2, 9))
        #random_a_bit = random.choice(np.arange(4, 9))
        random_w_bit = random.choice(np.arange(self.w_range[0], self.w_range[1]))
        random_a_bit = random.choice(np.arange(self.a_range[0], self.a_range[1]))
        self.config.defrost()
        self.config.QUANT.W.BIT = int(random_w_bit)
        self.config.QUANT.A.BIT = int(random_a_bit)

        # allocate bit number for encoder_q
        self.encoder_q.allocate_bit(self.config, None)
        self.encoder_q.set_quant_state(True, True, w_init=True, a_init=True)
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # compute logits for float
        l_pos_f = torch.einsum('nc,nc->n', [q_f, k]).unsqueeze(-1)
        l_neg_f = torch.einsum('nc,ck->nk', [q_f, self.queue.clone().detach()])
        logits_f = torch.cat([l_pos_f, l_neg_f], dim=1)
        logits_f /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return [logits, logits_f], labels
