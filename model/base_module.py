import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BaseModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def init_uniform(self, shape):
        if isinstance(shape, int):
            shape = [shape]
        stdv = 1. / math.sqrt(shape[-1])
        w = torch.empty(shape)
        nn.init.uniform_(w, -stdv, stdv)
        return w
    
    def average_pooling(self, h, mask, pred, succ):
        '''
        h: [batch_size, seq_len, embedding_dim], mask: [batch_size, seq_len]
        '''
        h_mean = []
        for i in range(h.size(0)):
            r = max(torch.sum(mask[i]).item(), pred+succ)
            m = torch.mean(h[i, pred:r-succ], 0)  # remove cls & sep
            h_mean.append(m)
        outs = torch.stack(h_mean)  # [batch_size, embedding_dim]
        return outs
    
    def forward(self, data):
        raise NotImplementedError