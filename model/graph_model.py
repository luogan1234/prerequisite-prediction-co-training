import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from model.gcn_layer import GCNLayer
from model.gat_layer import GatLayer
from model.mlp_layer import MLPLayer

class GraphModel(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.layer1 = self.new_layer(config.graph_layer, config.laplacian1)
        self.layer2 = self.new_layer(config.graph_layer, config.laplacian2)
        self.fc = MLPLayer(self.config.encoding_dim*2, self.config.num_classes)
        self.cached_embeds1 = torch.zeros(config.concept_num, config.encoding_dim).to(config.device)
        self.cached_embeds2 = torch.zeros(config.concept_num, config.encoding_dim).to(config.device)
    
    def new_layer(self, graph_layer, laplacian):
        if graph_layer == 'gcn':
            layer = GCNLayer(self.config, laplacian)
        if graph_layer == 'gat':
            layer = GatLayer(self.config, laplacian)
        assert layer, 'Graph layer is not supported.'
        return layer
    
    def forward(self, batch):
        i1, i2 = batch['i1'], batch['i2']
        o1 = self.layer1(self.config.embeddings, i1)
        o2 = self.layer2(self.config.embeddings, i2)
        o = torch.cat([o1, o2], -1)
        outs = self.fc(o)
        return outs
    
    def set_cached_embeds(self, batch):
        i = batch['i']
        o1 = self.layer1(self.config.embeddings, i)
        o2 = self.layer2(self.config.embeddings, i)
        pos = [i]
        self.cached_embeds1.index_put_(pos, o1.detach())
        self.cached_embeds2.index_put_(pos, o2.detach())
    
    def predict(self, batch):
        i1, i2 = batch['i1'], batch['i2']
        o1 = self.cached_embeds1.index_select(0, i1)
        o2 = self.cached_embeds2.index_select(0, i2)
        o = torch.cat([o1, o2], -1)
        outs = self.fc(o)
        return outs