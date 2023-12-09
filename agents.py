import numpy as np
import random
from time import time
from util import Directions


class Agent():
    def __init__(self, x_pos_init=0, y_pos_init=0):
        self.score = 0
        self.cumulative_score = 0
        self.x_pos = x_pos_init
        self.y_pos = y_pos_init
        self.moves = []
        self.is_in_game = True


    def step(self):
        return -1

    # return agent position in (i,j) convention
    def get_agent_pos(self,):
        return self.y_pos, self.x_pos

    def update_agent_pos(self, new_x_pos, new_y_pos):
        self.x_pos = new_x_pos
        self.y_pos = new_y_pos

    def agent_is_at_pos(self, x_pos, y_pos):
        return (self.x_pos == x_pos and self.y_pos == y_pos)
    
    def remove_agent_from_game(self,):
        self.is_in_game = False


class RandomAgent(Agent):

    def __init__(self, x_pos_init=0, y_pos_init=0):
        super().__init__(x_pos_init=x_pos_init, y_pos_init=y_pos_init)


    def choose_move(self, sandpile):

        possible_moves = [Directions.STAY, Directions.LEFT, Directions.RIGHT, Directions.UP, Directions.DOWN]

        if self.x_pos <= sandpile.left_bound_idx:
            possible_moves.remove(Directions.LEFT)

        if self.x_pos >= sandpile.right_bound_idx:
            possible_moves.remove(Directions.RIGHT)

        if self.y_pos <= sandpile.top_bound_idx:
            possible_moves.remove(Directions.UP)

        if self.y_pos >= sandpile.bot_bound_idx:
            possible_moves.remove(Directions.DOWN)

        print('possible_moves')
        print(possible_moves)
        move = random.choice(possible_moves)
        print('move')
        print(move)
        self.moves.append(move)
        
        return move
    
    

