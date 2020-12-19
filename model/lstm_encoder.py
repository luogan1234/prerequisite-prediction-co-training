import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.base_module import BaseModule

class LSTMEncoder(BaseModule):
    def __init__(self, config):
        super().__init__(config)
        self.word_embedding = nn.Embedding(self.config.vocab_num, self.config.word_embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.config.word_embedding_dim, self.config.text_embedding_dim, 1, batch_first=True, bidirectional=False)
    
    def forward(self, inputs):
        x = self.word_embedding(inputs)
        o, _ = self.lstm(x)  # [batch_size, seq_len, embedding_dim]
        outs = o[:, -1, :]
        return outs