import numpy as np
import random
from util import Directions

class Sandpile():

    def __init__(self,N_grid=2, MAXIMUM_GRAINS=4, agents=[], DROP_SAND=True, MAX_STEPS=1000):

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
        self.agents = agents
        self.REWARD_OFF_GRID = -1e4
        self.MAX_STEPS = MAX_STEPS
        self.iteration = 0

    def step(self,):
        #check that the game is still running
        game_is_running = self.iteration < self.MAX_STEPS

        for agent in self.agents:
            game_is_running = game_is_running and agent.is_in_game

        if not game_is_running:
            return game_is_running

        self.iteration += 1
        # determine if we should avalanche, based on if any of the grid values
        # are greater than the alloweable maximum grain number
        self.is_avalanching = np.any(self.grid >= self.MAXIMUM_GRAINS)

        
        # update the sandpile environment
        if not self.is_avalanching:
            # print('NOT AVALANCHING')
            if self.DROP_SAND:
                self.drop_sandgrain()

            # run the agents
            for agent in self.agents:
                if agent.is_in_game:
                    
                    
                    print('agent_pos (i,j) (Y, X): ', agent.get_agent_pos())

                    print('Moving agent')
                    # have the agent choose a direction to move in
                    direction = agent.choose_move(self)
                    self.move_agent_in_direction(direction, agent)

                    print('agent_pos after normal move (i,j) (Y, X): ', agent.get_agent_pos())

                    # check if agent is in game
                    if not self.check_agent_is_in_grid(agent):
                        agent.remove_agent_from_game()
                        print('REMOVING AGENT FROM GAME')
            
            # handle state transition from avalanching to not avalanching
            # this records the avalanche size
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
        for agent in self.agents:
            # if the agent is still in the game the reward is the value of the index
            if self.check_agent_is_in_grid(agent):
                y_pos, x_pos = agent.get_agent_pos()
                reward = self.grid[y_pos, x_pos]
                print('REWARD FOR MOVE: ', reward)
                agent.get_reward(reward)

            else:
                print('REWARD FOR OFF GRID')
                agent.get_reward(self.REWARD_OFF_GRID)

            # print('rewards: ', self.agent.rewards)
            # print('cumulative_score: ', self.agent.cumulative_score)
            # print('cumulative_rewards:', self.agent.cumulative_rewards)

        # input()
        return game_is_running

    def move_agent_to_point(self, agent, new_x_pos, new_y_pos):
        agent.update_agent_pos(new_x_pos, new_y_pos)
            
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
        self.move_agent_to_point(agent, new_x_pos, new_y_pos)


    def choose_random_neighbor_from_point(self, x_pos, y_pos):

        possible_moves = [Directions.LEFT, Directions.RIGHT, Directions.UP, Directions.DOWN]
        direction = random.choice(possible_moves)
        new_x_pos, new_y_pos = self.get_new_pos_from_direction(direction, x_pos, y_pos)

        return new_x_pos, new_y_pos

    def move_agent_random_from_point(self, agent):
        x_pos = agent.x_pos
        y_pos = agent.y_pos
        new_agent_x_pos, new_agent_y_pos = self.choose_random_neighbor_from_point(x_pos, y_pos)
        self.move_agent_to_point(agent, new_agent_x_pos, new_agent_y_pos)
    
    def check_agent_is_in_grid(self, agent):
        in_x_pos = agent.x_pos >= self.left_bound_idx and agent.x_pos <= self.right_bound_idx
        in_y_pos = agent.y_pos >= self.top_bound_idx and agent.y_pos <= self.bot_bound_idx

        return in_x_pos and in_y_pos

    def drop_sandgrain(self,):
        # drop a grain on a uniformly selected location
        # select a random location
        x_coord_grain = random.randint(self.left_bound_idx, self.right_bound_idx)
        y_coord_grain = random.randint(self.top_bound_idx, self.bot_bound_idx)

        # print('y_coord_grain, x_coord_grain')
        # print(y_coord_grain, x_coord_grain)

        #increment the count at that location
        self.grid[y_coord_grain, x_coord_grain] += 1
        # print(self.grid)

    def avalanche(self,):
        print('AVALANCHING')
        
        for agent in self.agents:
            print('agent_pos (i,j) (Y, X) before avalanche: ', agent.get_agent_pos())
            self.print_grid_and_agent_pos(agent)

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

        x_coord_unstable = rand_unstable[1]
        y_coord_unstable = rand_unstable[0]

        print('y_coord_unstable, x_coord_unstable')
        print(y_coord_unstable, x_coord_unstable)


        # topple the grid at this coordinate
        self.grid[y_coord_unstable, x_coord_unstable] -= self.MAXIMUM_GRAINS

        # increment neighboring vertex counts
        self.increment_neighbors(y_coord_unstable, x_coord_unstable)

        print('POST TOPPLE')
        self.print_grid()

        # move the agent to one of the neighbors if the agent was at the unstable coordinate
        for agent in self.agents:
            # check if agent is at the coordinate
            if agent.is_in_game:
                agent_is_at_unstable_pos = agent.agent_is_at_pos(x_coord_unstable, y_coord_unstable)

            if agent.is_in_game and agent_is_at_unstable_pos:
                
                print('moving agent due to avalanche')
                self.move_agent_random_from_point(agent)
                
                print('agent pos after avalanche (i,j) (Y, X)', agent.get_agent_pos())
                self.print_agent_pos_on_grid(agent)
                # input()
        # input()

        
        
        # check if agent is still in game
        for agent in self.agents:
            if not self.check_agent_is_in_grid(agent):
                agent.remove_agent_from_game()
                print('REMOVING AGENT FROM GAME FROM AVALANCHE')


       
        # print(self.grid)

        # input()

    def increment_neighbors(self, y_coord, x_coord):

        if (x_coord - 1) >= self.left_bound_idx :
            self.grid[y_coord, x_coord - 1] += 1

        if (x_coord + 1) <= self.right_bound_idx:
            self.grid[y_coord, x_coord + 1] += 1
        
        if (y_coord - 1) >= self.top_bound_idx :
            self.grid[y_coord - 1, x_coord] += 1

        if (y_coord + 1) <= self.bot_bound_idx:
            self.grid[y_coord + 1, x_coord] += 1

    #returns the part of the grid that contains the sand
    # def get_sandpile(self,):
    #     return self.grid[1:self.N_grid+1, 1:self.N_grid+1]

    def get_sandpile(self,):
        return self.grid


    def print_grid(self,):
        print(self.get_sandpile())

    def get_agent_pos_on_grid(self, agent):
        #generate a grid of zeros with a 1 at the agent pos
        agent_grid = np.zeros((self.N_grid, self.N_grid))
        y_pos, x_pos = agent.get_agent_pos()
        if self.check_agent_is_in_grid(agent):
            agent_grid[y_pos, x_pos] = 1
        return agent_grid


    def print_agent_pos_on_grid(self, agent):
        agent_grid = self.get_agent_pos_on_grid(agent)
        print(agent_grid)

    def print_grid_and_agent_pos(self, agent):
        agent_grid = self.get_agent_pos_on_grid(agent)
        print(self.get_sandpile(),'\n\n' , agent_grid)

    def add_agent(self, agent):
        self.agents.append(agent) 
        
