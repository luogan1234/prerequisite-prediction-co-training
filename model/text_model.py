import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from model.lstm_encoder import LSTMEncoder
from model.bert_encoder import BERTEncoder
from transformers import BertModel
from model.mlp_layer import MLPLayer

class TextModel(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        if config.text_encoder == 'lstm':
            self.encoder = LSTMEncoder(config)
        if config.text_encoder in ['bert', 'bert-freeze']:
            self.encoder = BERTEncoder(config)
        self.fc = MLPLayer(self.config.text_embedding_dim*2, self.config.num_classes)
        self.cached_embeds = torch.zeros(config.concept_num, config.text_embedding_dim).to(config.device)
    
    def get_embeddings(self, batch):
        x = batch['t']
        outs = self.encoder(x)
        return outs
    
    def forward(self, batch):
        t1, t2 = batch['t1'], batch['t2']
        e1, e2 = self.encoder(t1), self.encoder(t2)
        o = torch.cat([e1, e2], 1)
        outs = self.fc(o)
        return outs
    
    def set_cached_embeds(self, batch):
        i, t = batch['i'], batch['t']
        e = self.encoder(t)
        pos = [i]
        self.cached_embeds.index_put_(pos, e.detach())
    
    def predict(self, batch):
        i1, i2 = batch['i1'], batch['i2']
        e1 = self.cached_embeds.index_select(0, i1)
        e2 = self.cached_embeds.index_select(0, i2)
        o = torch.cat([e1, e2], -1)
        outs = self.fc(o)
        return outs