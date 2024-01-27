import torch
import torch.nn as nn 
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from agents import Agent
from util import Directions

class GenericRLAgent(nn.Module, Agent):
    """Implementation for a Generic Reinforcement Learning Agent
    Inherits from base Agent class to ensure functionality with 
    the sandpile mechanics. Currently a standard MLP.

    """
    def __init__(self, input_dim, num_hidden_layers, hidden_dim, output_dim, device, x_pos_init=0, y_pos_init=0):
        """
        Args:
            input_dim (int): dimension of inputs
            num_hidden_layers (int): number of hidden layers
            hidden_dim (int): dimension of hidden layers
            output_dim (int): dimension of output
            device: which torch device this agent will live on
            x_pos_init (int, optional): initial x agent position (j) on sandpile. Defaults to 0.
            y_pos_init (int, optional): initial y agent position (i) on sandpile. Defaults to 0.
        """
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
            
        output = self.output_layer(x)
        return output
        

    def select_action(self, sandpile, x_pos, y_pos):
        """
        #x_pos = j
        #y_pos = i

        """
        
        # normalize sandpile to [0,1]
        N_grid = sandpile.N_grid
        sandpile_norm = normalize_sandpile_grid(sandpile)

        # append -1 values around the sandpile at the agent's current location such that the agent will always
        # receive an (2*N_grid-1) x (2*N_grid-1) of information
        grid_input = pad_sandpile_grid_with_void(sandpile_norm, N_grid, x_pos, y_pos)

        input_np = grid_input.reshape(-1)
        sandpile_tensor = torch.from_numpy(input_np).float().to(self.device)
        
        # sample action
        logits = self.forward(sandpile_tensor)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropies_actions = -(log_probs * probs).sum()
        m = Categorical(probs)
        action = m.sample()

        # store values
        self.action_idx = action.item()
        self.log_prob = m.log_prob(action)
        self.entropy = entropies_actions
        
        return self.action_idx, self.log_prob, entropies_actions

    def choose_move(self, sandpile):
        """Choose explicit direction for sandpile
        """
        action_idx, log_prob, entropies_actions = self.select_action(sandpile, self.x_pos, self.y_pos)
        move = list(Directions)[action_idx]
        self.moves.append(move)
        
        return move

    def _test_counter(self,):
        self._test_counter_i += 1


def normalize_sandpile_grid(sandpile):
    """Normalize sandpile to [0,1] to stabilize training
    """
    sandpile_norm = (sandpile.grid - (0)) * (1 - 0)/ ((sandpile.MAXIMUM_GRAINS-1) - 0) + (0)

    return sandpile_norm

def pad_sandpile_grid_with_void(sandpile_grid, N_grid, x_pos, y_pos):
    """Pads the sandpile environment with -1's at the specified position
    (typically the agent's position) such that the output is (2*N_grid-1) x (2*N_grid-1)
    """
    void_grid = -1 * np.ones((N_grid,N_grid))

    super_grid = np.block([[void_grid, void_grid, void_grid],
                            [void_grid, sandpile_grid, void_grid],
                            [void_grid, void_grid, void_grid]])

    # compute x_pos, y_pos in new grid coordinates
    x_pos_s = x_pos + N_grid
    y_pos_s = y_pos + N_grid

    padded_grid = super_grid[(y_pos_s - N_grid + 1):(y_pos_s+N_grid ), (x_pos_s - N_grid + 1):(x_pos_s+N_grid )]

    return padded_grid

class Policy(GenericRLAgent):
    def __init__(self, input_dim, num_hidden_layers, hidden_dim, output_dim, device, x_pos_init=0, y_pos_init=0):
        super().__init__(input_dim=input_dim, 
                        num_hidden_layers=num_hidden_layers, 
                        hidden_dim=hidden_dim, 
                        output_dim=output_dim, 
                        device=device, 
                        x_pos_init=x_pos_init, 
                        y_pos_init=y_pos_init)
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
            
        action_scores = self.output_layer(x)
        return action_scores
        
class ActorCritic(GenericRLAgent):
    """Implementes Actor Critic policy

    """
    def __init__(self, input_dim, num_hidden_layers, hidden_dim, output_dim, device, x_pos_init=0, y_pos_init=0):
        super().__init__(input_dim=input_dim, 
                        num_hidden_layers=num_hidden_layers, 
                        hidden_dim=hidden_dim, 
                        output_dim=output_dim, 
                        device=device, 
                        x_pos_init=x_pos_init, 
                        y_pos_init=y_pos_init)

        self.critic_output = nn.Linear(hidden_dim, 1)
        
        
    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)


        # generate action logits for actor
        action_scores = self.output_layer(x)

        # generate value for critic
        critic_score = self.critic_output(x)
        return action_scores, critic_score
    
    def select_action(self, sandpile, x_pos, y_pos):
        """
        #x_pos = j
        #y_pos = i

        """
        
        # normalize sandpile to [0,1]
        N_grid = sandpile.N_grid
        sandpile_norm = normalize_sandpile_grid(sandpile)

        # append -1 values around the sandpile at the agent's current location such that the agent will always
        # receive an (2*N_grid-1) x (2*N_grid-1) of information
        grid_input = pad_sandpile_grid_with_void(sandpile_norm, N_grid, x_pos, y_pos)

        input_np = grid_input.reshape(-1)
        sandpile_tensor = torch.from_numpy(input_np).float().to(self.device)
        
        # sample action and generate critic score 
        logits, critic_score = self.forward(sandpile_tensor)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropies_actions = -(log_probs * probs).sum()
        m = Categorical(probs)
        action = m.sample()

        # store values
        self.action_idx = action.item()
        self.log_prob = m.log_prob(action)
        self.entropy = entropies_actions
        self.value = critic_score
        return self.action_idx, self.log_prob, entropies_actions

    def choose_move(self, sandpile):
        action_idx, critic_score, log_prob, entropies_actions = self.select_action(sandpile, self.x_pos, self.y_pos)
        move = list(Directions)[action_idx]
        self.moves.append(move)
        
        return move
        