import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.base_module import BaseModule

class GatLayer(BaseModule):
    def __init__(self, config, laplacian):
        super().__init__(config)
        self.laplacian = laplacian
        self.Wq = nn.Linear(config.graph_embedding_dim, config.attention_dim)
        self.Wk = nn.Linear(config.graph_embedding_dim, config.attention_dim)
        self.Wv = nn.Linear(config.graph_embedding_dim, config.encoding_dim)
    
    def forward(self, embeddings, indexes):
        alphas = torch.mm(self.Wq(embeddings), self.Wk(embeddings).transpose(0, 1))
        alphas = F.sigmoid(alphas)*self.laplacian
        x = torch.mm(alphas, self.Wv(embeddings))
        outs = x.index_select(0, indexes)
        outs = torch.relu(outs)
        return outs