import numpy as np
import random
from time import time
from util import Directions, get_new_pos_from_direction, choose_random_neighbor_from_point, calculate_best_move_to_reach_pos


class Agent():
    """This class implements the base agent that operates in the sandpile
    """
    def __init__(self, x_pos_init=0, y_pos_init=0):
        self.score = 0
        self.cumulative_score = 0
        self.x_pos_init = x_pos_init
        self.y_pos_init = y_pos_init
        self.x_pos = x_pos_init
        self.y_pos = y_pos_init
        self.moves = []
        self.rewards = []
        self.cumulative_rewards = []
        self.in_game = True
        self.is_getting_avalanched = False
        self.was_moved_during_avalanche_step = False


    def choose_move(self, sandpile):
        return Directions.STAY

    # return agent position in (i,j) convention
    def get_agent_pos(self,):
        return self.y_pos, self.x_pos

    def move_agent_to_point(self, new_x_pos, new_y_pos):
        self.x_pos = new_x_pos
        self.y_pos = new_y_pos

    def agent_is_at_pos(self, x_pos, y_pos):
        return (self.x_pos == x_pos and self.y_pos == y_pos)
    
    def is_in_game(self,):
        return self.in_game

    def remove_agent_from_game(self,):
        self.in_game = False
    
    def move_agent_in_direction(self, direction):
        
        new_x_pos, new_y_pos = get_new_pos_from_direction(direction, self.x_pos, self.y_pos)
        self.move_agent_to_point(new_x_pos, new_y_pos)
    

    def move_agent_random_from_point(self, ):
        x_pos = self.x_pos
        y_pos = self.y_pos
        new_agent_x_pos, new_agent_y_pos = choose_random_neighbor_from_point(x_pos, y_pos)
        self.move_agent_to_point(new_agent_x_pos, new_agent_y_pos)

    
    # add reward to list of rewards for each step
    def append_reward(self, reward):
        self.rewards.append(reward)
        self.cumulative_rewards = np.cumsum(np.array(self.rewards))
        self.cumulative_score = np.sum(self.rewards)

    def get_cumulative_score(self,):
        return self.cumulative_score
    

    # get possible moves: the agent will never choose to fall off
    # the edge of the board
    def get_possible_moves_stay_in_grid(self, sandpile):
        possible_moves = list(Directions)

        if self.x_pos <= sandpile.left_bound_idx:
            possible_moves.remove(Directions.LEFT)

        if self.x_pos >= sandpile.right_bound_idx:
            possible_moves.remove(Directions.RIGHT)

        if self.y_pos <= sandpile.top_bound_idx:
            possible_moves.remove(Directions.UP)

        if self.y_pos >= sandpile.bot_bound_idx:
            possible_moves.remove(Directions.DOWN)

        return possible_moves
    
    def get_is_getting_avalanched(self,):
        return self.is_getting_avalanched

    def set_is_getting_avalanched(self, is_getting_avalanched):
        self.is_getting_avalanched = is_getting_avalanched

    def get_was_moved_during_avalanche_step(self,):
        return self.was_moved_during_avalanche_step

    def set_was_moved_during_avalanche_step(self, was_moved):
        self.was_moved_during_avalanche_step = was_moved

    def reset(self,):
        self.score = 0
        self.cumulative_score = 0
        self.x_pos = self.x_pos_init
        self.y_pos = self.y_pos_init
        self.moves = []
        self.rewards = []
        self.cumulative_rewards = []
        self.in_game = True
        self.is_getting_avalanched = False
        self.was_moved_during_avalanche_step = False


class RandomAgent(Agent):
    """This agent will choose moves randomly but will 
    not choose a move that removes it from the grid
    """
    def __init__(self, x_pos_init=0, y_pos_init=0):
        super().__init__(x_pos_init=x_pos_init, y_pos_init=y_pos_init)


    def choose_move(self, sandpile):
        """Choose a move randomly
        """
        possible_moves = self.get_possible_moves_stay_in_grid(sandpile)
        move = random.choice(possible_moves)
        self.moves.append(move)
        
        return move
    
    
class MaxAgent(Agent):
    """ This agent chooses to move to the nearest square (given the four cardinal directions
     plus staying in place) that has the highest score
    """
    

    def __init__(self, x_pos_init=0, y_pos_init=0):
        super().__init__(x_pos_init=x_pos_init, y_pos_init=y_pos_init)

    def choose_move(self, sandpile):
        
        possible_moves = self.get_possible_moves_stay_in_grid(sandpile)

        # shuffle possible moves, this is done because in the event of 
        # an eventual tie between the maximums, numpy only returns the 
        # first index, so we want to avoid bias based on which Direction
        # is listed first
        random.shuffle(possible_moves)

        # evaluate possible moves based on maximum reward
        # add all moves that have a maximum reward
        prewards = []
        for pmove in possible_moves:
            new_x_pos, new_y_pos = get_new_pos_from_direction(pmove, self.x_pos, self.y_pos)
            preward = sandpile.grid[new_y_pos, new_x_pos]
            prewards.append(preward)


        # print('prewards')
        # print(prewards)
        max_reward_idx = np.argmax(prewards)
        # print('max_reward_idx')
        # print(max_reward_idx)
        move = possible_moves[max_reward_idx]
        self.moves.append(move)

        return move

class SeekSpecificValueAgent(Agent):
    """ This agent chooses to move to the nearest square (given the four cardinal directions
     plus staying in place) that has a specific value
    """
    

    def __init__(self, x_pos_init=0, y_pos_init=0, specific_value=1):
        super().__init__(x_pos_init=x_pos_init, y_pos_init=y_pos_init)
        self.specific_value = specific_value

    def choose_move(self, sandpile):
        # chooses the move to go to a specific value
        # if it can't find the value, choose a random direction
        
        possible_moves = self.get_possible_moves_stay_in_grid(sandpile)

        # shuffle possible moves, this is done because in the event of 
        # an eventual tie between the specific values, numpy only returns the 
        # first index, so we want to avoid bias based on which Direction
        # is listed first
        random.shuffle(possible_moves)

        # evaluate possible moves based on targeted value
        moves_to_value = []
        for pmove in possible_moves:
            new_x_pos, new_y_pos = get_new_pos_from_direction(pmove, self.x_pos, self.y_pos)
            preward = sandpile.grid[new_y_pos, new_x_pos]

            if preward == self.specific_value:
                moves_to_value.append(pmove)

        if len(moves_to_value) > 0:
            move = moves_to_value[0]
        else:
            move = random.choice(possible_moves)

        self.moves.append(move)
        return move

class SeekCenterAgent(Agent):
    """ This agent chooses to move to the center of the sandpile
    """
   

    def __init__(self, x_pos_init=0, y_pos_init=0):
        super().__init__(x_pos_init=x_pos_init, y_pos_init=y_pos_init)

    def choose_move(self, sandpile):
        # chooses the move to go to the center
        
        possible_moves = self.get_possible_moves_stay_in_grid(sandpile)

        # shuffle possible moves, this is done because in the event of 
        # an eventual tie between the specific values, numpy only returns the 
        # first index, so we want to avoid bias based on which Direction
        # is listed first
        random.shuffle(possible_moves)

        # compute center position
        center_pos = ((sandpile.N_grid-1)//2, (sandpile.N_grid-1)//2)

        # find the best move to reach the center position
        best_move = calculate_best_move_to_reach_pos(center_pos[0], center_pos[1], self.x_pos, self.y_pos)

        self.moves.append(best_move)
        return best_move
