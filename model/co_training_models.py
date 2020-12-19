import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule
from model.text_model import TextModel
from model.graph_model import GraphModel

class CoTrainingModels(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.text_models = []
        self.graph_models = []
    
    def new_text_model(self):
        model = TextModel(self.config)
        self.text_models.append(model)
        return model
    
    def new_graph_model(self):
        model = GraphModel(self.config)
        self.graph_models.append(model)
        return model
    
    def clear_models(self, mode=''):
        assert mode in ['text', 'graph', '']
        if mode in ['text', '']:
            self.text_models = []
        if mode in ['graph', '']:
            self.graph_models = []
    
    def load_state_dict(self, para_list, mode):
        assert mode in ['text', 'graph']
        if mode == 'text':
            new_model = self.new_text_model
        if mode == 'graph':
            new_model = self.new_graph_model
        for para in para_list:
            model = new_model()
            model.load_state_dict(para)
        print('Load total {} {} models.'.format(len(para_list), mode))
    
    def state_dict(self, mode):
        assert mode in ['text', 'graph']
        if mode == 'text':
            models = self.text_models
        if mode == 'graph':
            models = self.graph_models
        para_list = []
        for model in models:
            para_list.append(model.state_dict())
        return para_list
    
    def get_text_embeddings(self, batch):
        embeds = []
        for model in self.text_models:
            embeds.append(model.get_embeddings(batch))
        embeds = torch.mean(torch.stack(embeds), 0)
        return embeds
    
    def train(self, mode=''):
        assert mode in ['text', 'graph', '']
        models = []
        if mode in ['text', '']:
            models += self.text_models
        if mode in ['graph', '']:
            models += self.graph_models
        for model in models:
            model.train()
    
    def eval(self, mode=''):
        assert mode in ['text', 'graph', '']
        models = []
        if mode in ['text', '']:
            models += self.text_models
        if mode in ['graph', '']:
            models += self.graph_models
        for model in models:
            model.eval()
    
    def to(self, device, mode=''):
        assert mode in ['text', 'graph', '']
        models = []
        if mode in ['text', '']:
            models += self.text_models
        if mode in ['graph', '']:
            models += self.graph_models
        for model in models:
            model.to(device)
    
    def forward(self, batch, mode=''):
        assert mode in ['text', 'graph', '']
        models = []
        if mode in ['text', '']:
            models += self.text_models
        if mode in ['graph', '']:
            models += self.graph_models
        outs = []
        for model in models:
            outs.append(F.softmax(model(batch), -1))
        outs = torch.mean(torch.stack(outs), 0)
        return outs