import torch
import torch.nn as nn 
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from agents import Agent
from util import Directions

class Policy(nn.Module, Agent):
    def __init__(self, input_dim, num_hidden_layers, hidden_dim, output_dim, device, x_pos_init=0, y_pos_init=0):
        super().__init__()
        Agent.__init__(self, x_pos_init=x_pos_init, y_pos_init=y_pos_init)
        self.device = device
        
        self.input_dim = input_dim
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
        self.hidden_layers.append(nn.LeakyReLU(0.1))
        self.hidden_layers.append(nn.Dropout(p=0.2))
        
        for _ in range(0, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.hidden_layers.append(nn.LeakyReLU(0.1))
            self.hidden_layers.append(nn.Dropout(p=0.2))
            
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._test_counter_i = 0
        self.log_prob = None
        self.action_idx = None
        self.entropy = None

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            
        action_scores = self.output_layer(x)
        return action_scores
        # return F.softmax(, dim=-1)
        

    def select_action(self, sandpile, x_pos, y_pos):
        #x_pos = j
        #y_pos = i
        # normalize sandpile to [0,1]
        N_grid = sandpile.N_grid
        sandpile_norm = (sandpile.grid - (0)) * (1 - 0)/ ((sandpile.MAXIMUM_GRAINS-1) - 0) + (0)
        void_grid = -1 * np.ones(sandpile.grid.shape)

        super_grid = np.block([[void_grid, void_grid, void_grid],
                                [void_grid, sandpile_norm, void_grid],
                                [void_grid, void_grid, void_grid]])

        # print('super_grid')
        # print(super_grid)

        # compute x_pos, y_pos in new grid coordinates
        x_pos_s = x_pos + N_grid
        y_pos_s = y_pos + N_grid

        grid_input = super_grid[(y_pos_s - N_grid + 1):(y_pos_s+N_grid ), (x_pos_s - N_grid + 1):(x_pos_s+N_grid )]
        # print('grid_input')
        # print(grid_input)

        # print(x_pos, y_pos, sandpile.N_grid//2)
        # pos_norm = (np.array([x_pos, y_pos]) - (0)) * (1 - -1)/ ((sandpile.N_grid-1) - 0) + (-1)
        # input_np = np.concatenate((sandpile_input_norm, pos_norm))

        input_np = grid_input.reshape(-1)
        sandpile_tensor = torch.from_numpy(input_np).float().to(self.device)
        
        logits = self.forward(sandpile_tensor)
        # print('logits ', logits)
        probs = F.softmax(logits, dim=-1)
        # print('probs: ', probs)
        # print('probs grad', probs.grad)
        log_probs = F.log_softmax(logits, dim=-1)
        entropies_actions = -(log_probs * probs).sum()
        m = Categorical(probs)
        # print('m: ', m)

        action = m.sample()
        # print('action: ', action)
        return action.item(), m.log_prob(action), entropies_actions

    def choose_move(self, sandpile):
        action_idx, log_prob, entropies_actions = self.select_action(sandpile, self.x_pos, self.y_pos)
        self.action_idx = action_idx
        self.log_prob = log_prob
        self.entropy = entropies_actions
        move = list(Directions)[action_idx]
        self.moves.append(move)
        
        return move

    def _test_counter(self,):
        self._test_counter_i += 1