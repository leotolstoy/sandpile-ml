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
        self.rewards = []
        self.cumulative_rewards = []
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
    
    # add reward to list of rewards
    def get_reward(self, reward):
        self.rewards.append(reward)
        self.cumulative_rewards = np.cumsum(np.array(self.rewards))
        self.cumulative_score = np.sum(self.rewards)

    # get possible moves: the agent will never choose to fall off
    # the edge of the board
    def get_possible_moves(self, sandpile):
        possible_moves = [Directions.STAY, Directions.LEFT, Directions.RIGHT, Directions.UP, Directions.DOWN]

        if self.x_pos <= sandpile.left_bound_idx:
            possible_moves.remove(Directions.LEFT)

        if self.x_pos >= sandpile.right_bound_idx:
            possible_moves.remove(Directions.RIGHT)

        if self.y_pos <= sandpile.top_bound_idx:
            possible_moves.remove(Directions.UP)

        if self.y_pos >= sandpile.bot_bound_idx:
            possible_moves.remove(Directions.DOWN)

        return possible_moves



class RandomAgent(Agent):

    def __init__(self, x_pos_init=0, y_pos_init=0):
        super().__init__(x_pos_init=x_pos_init, y_pos_init=y_pos_init)


    def choose_move(self, sandpile):

        possible_moves = self.get_possible_moves(sandpile)

        print('possible_moves')
        print(possible_moves)
        move = random.choice(possible_moves)
        print('move')
        print(move)
        self.moves.append(move)
        
        return move
    
    
class MaxAgent(Agent):
    # This agent chooses to move to the nearest square (given the four cardinal directions
    # plus staying in place) that has the highest score

    def __init__(self, x_pos_init=0, y_pos_init=0):
        super().__init__(x_pos_init=x_pos_init, y_pos_init=y_pos_init)

    def choose_move(self, sandpile):
        
        possible_moves = self.get_possible_moves(sandpile)

        max_reward = 0
        moves_with_highest_reward = []

        # evaluate possible moves based on maximum reward
        # add all moves that have a maximum reward
        prewards = []
        for pmove in possible_moves:
            new_x_pos, new_y_pos = sandpile.get_new_pos_from_direction(pmove, self.x_pos, self.y_pos)
            preward = sandpile.grid[new_y_pos, new_x_pos]
            # print('preward')
            # print(preward)
            prewards.append(preward)
            # if preward >= max_reward:
            #     print('moves_with_highest_reward')
            #     print(moves_with_highest_reward)
            #     max_reward = preward
            #     moves_with_highest_reward.append(pmove)


        print('prewards')
        print(prewards)
        max_reward_idx = np.argmax(prewards)
        print('max_reward_idx')
        print(max_reward_idx)
        move = possible_moves[max_reward_idx]

        #choose randomly from the moves that have maximum reward
        # typically moves_with_highest_reward will only have a single element
        # print('moves_with_highest_reward')
        # print(moves_with_highest_reward)
        # move = random.choice(moves_with_highest_reward)

        return move