import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        mid_dim = in_dim // 2
        self.out_dim = out_dim
        self.fc1 = nn.Linear(in_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        input = self.dropout(input)
        x = F.relu(self.fc1(input))
        x = self.dropout(x)
        x = self.fc2(x)
        return x