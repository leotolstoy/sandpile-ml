import torch
import torch.nn as nn 
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_dim, output_dim):
        super(Policy, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        self.hidden_layers.append(nn.GELU())
        self.hidden_layers.append(nn.Dropout(p=0.2))
        
        for _ in range(0, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.GELU())
            self.hidden_layers.append(nn.Dropout(p=0.2))
            
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            
        action_scores = self.output_layer(x)
        return F.softmax(action_scores, dim=1)
        

    def select_action(self, sandpile):
        sandpile_tensor = torch.from_numpy(sandpile).reshape(-1).float()
        probs = self.forward(sandpile_tensor).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action
                                         )