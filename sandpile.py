import numpy as np
import random
from util import Directions

class Sandpile():

    def __init__(self,N_grid=2, MAXIMUM_GRAINS=4, agent=None, DROP_SAND=True):

        # set up the total grid, which is the sandpile plus empty void around it

        # X = N_grid x N_grid sandpile which gets sand dropped on it
        # O =  void which has a massively negative score
        # grid = 
        # O O O 
        # O X O
        # O O O
        # self.grid = np.ones((N_grid + 2, N_grid + 2)) * -1e3
        # self.grid[1:N_grid+1, 1:N_grid+1] = np.zeros((N_grid, N_grid))


        self.grid = np.zeros((N_grid, N_grid))

        self.N_grid = N_grid
        # self.left_bound_idx = 0 + 1
        # self.right_bound_idx = N_grid - 1 + 1
        # self.top_bound_idx = 0 + 1
        # self.bot_bound_idx = N_grid - 1 + 1

        self.left_bound_idx = 0
        self.right_bound_idx = N_grid - 1
        self.top_bound_idx = 0
        self.bot_bound_idx = N_grid - 1


        self.MAXIMUM_GRAINS = MAXIMUM_GRAINS
        self.DROP_SAND = DROP_SAND
        self.avalanche_size = 0
        self.is_avalanching = False
        self.was_avalanching_before = False
        self.avalanche_sizes = []
        self.agent = agent

    def step(self,):
        # determine if we should avalanche, based on if any of the grid values
        # are greater than the alloweable maximum grain number
        self.is_avalanching = np.any(self.grid >= self.MAXIMUM_GRAINS)

        # run the agent
        if self.agent and self.agent.is_in_game:

            # have the agent choose a direction to move in
            direction = self.agent.choose_move()
            self.move_agent_in_direction(direction, self.agent)


        # update the sandpile environment
        if not self.is_avalanching:
            if self.DROP_SAND:
                self.drop_sandgrain()
            
            if self.was_avalanching_before:
                self.avalanche_sizes.append(self.avalanche_size)
                self.was_avalanching_before = False
                self.avalanche_size = 0
                print(self.avalanche_sizes)


        else:
            self.avalanche()
            self.avalanche_size += 1
            self.was_avalanching_before = True

        # get agent rewards based on position

    def move_agent_to_point(self,new_x_pos, new_y_pos):
        self.agent.update_agent_pos(new_x_pos, new_y_pos)
            
    def get_new_pos_from_direction(self, direction, x_pos, y_pos):
        if direction == Directions.LEFT:
            new_x_pos = x_pos - 1
            new_y_pos = y_pos

        elif direction == Directions.RIGHT:
            new_x_pos = x_pos + 1
            new_y_pos = y_pos

        elif direction == Directions.UP:
            new_x_pos = x_pos
            new_y_pos = y_pos - 1

        elif direction == Directions.DOWN:
            new_x_pos = x_pos
            new_y_pos = y_pos + 1

        elif direction == Directions.STAY:
            new_x_pos = x_pos
            new_y_pos = y_pos

        return new_x_pos, new_y_pos


    def move_agent_in_direction(self, direction, agent):
        
        new_x_pos, new_y_pos = self.get_new_pos_from_direction(direction, agent.x_pos, agent.y_pos)
        self.move_agent_to_point(new_x_pos, new_y_pos)

    # def get_neighbor_idxs(x_pos, y_pos):

    def choose_random_neighbor_from_point(self, x_pos, y_pos):

        possible_moves = [Directions.LEFT, Directions.RIGHT, Directions.UP, Directions.DOWN]
        direction = random.choice(possible_moves)
        new_x_pos, new_y_pos = self.get_new_pos_from_direction(direction, x_pos, y_pos)

        return new_x_pos, new_y_pos

    def move_agent_random_from_point(self,):
        x_pos = self.agent.x_pos
        y_pos = self.agent.y_pos
        new_agent_x_pos, new_agent_y_pos = self.choose_random_neighbor_from_point(x_pos, y_pos)
        self.move_agent_to_point(new_agent_x_pos, new_agent_y_pos)
        

    def drop_sandgrain(self,):
        # drop a grain on a uniformly selected location
        # select a random location
        x_coord_grain = random.randint(self.left_bound_idx, self.right_bound_idx)
        y_coord_grain = random.randint(self.top_bound_idx, self.bot_bound_idx)

        # print(x_coord_grain, y_coord_grain)

        #increment the count at that location
        self.grid[x_coord_grain, y_coord_grain] += 1
        # print(self.grid)

    def avalanche(self,):
        print('AVALANCHING')
        self.print_grid()
        # find indices where avalanching/unstable
        # returns a Nx2 array of xy coordinates where N is the number of indices over the maximum
        avalanche_idxs = np.argwhere(self.grid >= self.MAXIMUM_GRAINS)
        N_avalanche_ixs = avalanche_idxs.shape[0]

        # print(avalanche_idxs)

        #pick an unstable vertex at random
        rand_idx = np.random.randint(N_avalanche_ixs)
        # print(rand_idx)
        rand_unstable = avalanche_idxs[rand_idx,:]
        # print(rand_unstable)

        x_coord_unstable = rand_unstable[0]
        y_coord_unstable = rand_unstable[1]

        # print(x_coord_unstable, y_coord_unstable)

        # check if agent is at the coordinate
        if self.agent and self.agent.is_in_game:
            agent_is_at_unstable_pos = self.agent.agent_is_at_pos(x_coord_unstable, y_coord_unstable)

        # topple the grid at this coordinate
        self.grid[x_coord_unstable, y_coord_unstable] -= self.MAXIMUM_GRAINS

        # increment neighboring vertex counts
        self.increment_neighbors(x_coord_unstable, y_coord_unstable)

        # move the agent to one of the neighbors if the agent was at the unstable coordinate
        if self.agent and self.agent.is_in_game and agent_is_at_unstable_pos:
            self.move_agent_random_from_point()


        print('POST TOPPLE')
        self.print_grid()
        # print(self.grid)

        # input()

    def increment_neighbors(self, x_coord, y_coord):

        if (x_coord - 1) >= self.left_bound_idx :
            self.grid[x_coord - 1, y_coord] += 1

        if (x_coord + 1) <= self.right_bound_idx:
            self.grid[x_coord + 1, y_coord] += 1
        
        if (y_coord - 1) >= self.top_bound_idx :
            self.grid[x_coord, y_coord - 1] += 1

        if (y_coord + 1) <= self.bot_bound_idx:
            self.grid[x_coord, y_coord + 1] += 1

    #returns the part of the grid that contains the sand
    # def get_sandpile(self,):
    #     return self.grid[1:self.N_grid+1, 1:self.N_grid+1]

    def get_sandpile(self,):
        return self.grid


    def print_grid(self,):
        print(self.get_sandpile())

