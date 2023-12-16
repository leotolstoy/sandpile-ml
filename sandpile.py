import numpy as np
import random

class Sandpile():

    def __init__(self,N_grid=2, initial_grid=None, MAXIMUM_GRAINS=4, agents=[], DROP_SAND=True, MAX_STEPS=1000):

        # allow for initial grid configuration
        if initial_grid is not None:
            self.grid = initial_grid
            assert(initial_grid.shape[0]==N_grid and initial_grid.shape[1]==N_grid)
        else:
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
        self.MAX_STEPS = MAX_STEPS
        self.iteration = 0

    def step(self,):
        #check that the game is still running
        iterations_not_exceeded = self.iteration < self.MAX_STEPS

        # establish that at least one agent is in the game, or if no agents were initialized
        # set this to true so that we can keep looping
        at_least_one_agent_in_game = False or len(self.agents) == 0
        for agent in self.agents:
            at_least_one_agent_in_game = at_least_one_agent_in_game or agent.is_in_game()

        game_is_running = iterations_not_exceeded and at_least_one_agent_in_game
        if not game_is_running:
            return game_is_running

        self.iteration += 1
        self.avalanche_size = 0

        agent_rewards = []

        # run the agents
        for agent in self.agents:
            if agent.is_in_game():
                
                # print('agent_pos (i,j) (Y, X): ', agent.get_agent_pos())
                # print('Moving agent')
                # have the agent choose a direction to move in
                direction = agent.choose_move(self)
                agent.move_agent_in_direction(direction)
                agent.set_is_getting_avalanched(False)

        if self.DROP_SAND:
            self.drop_sandgrain_randomly()

        #update avalanchine state
        self.is_avalanching = np.any(self.grid >= self.MAXIMUM_GRAINS)

        # avalanche 
        while self.is_avalanching:
            self.avalanche()
            self.avalanche_size += 1

        # update avalanche sizes
        if self.avalanche_size > 0:
            self.avalanche_sizes.append(self.avalanche_size)
            # print(self.avalanche_sizes)

        # update agent rewards
        for agent in self.agents:
            if agent.is_in_game() and not agent.get_is_getting_avalanched():
                y_pos, x_pos = agent.get_agent_pos()
                reward = self.grid[y_pos, x_pos]
                # print('REWARD FOR MOVE: ', reward)
                agent.append_reward(reward)
                agent_rewards.append(reward)
            else:
                agent_rewards.append(-100)

        # input()
        return self.grid, agent_rewards, game_is_running
    
    
    def check_agent_is_in_grid(self, agent):
        in_x_pos = agent.x_pos >= self.left_bound_idx and agent.x_pos <= self.right_bound_idx
        in_y_pos = agent.y_pos >= self.top_bound_idx and agent.y_pos <= self.bot_bound_idx

        return in_x_pos and in_y_pos

    def drop_sandgrain_at_pos(self, x_pos, y_pos):
        self.grid[y_pos, x_pos] += 1

    def drop_sandgrain_randomly(self,):
        # drop a grain on a uniformly selected location
        # select a random location
        x_coord_grain = random.randint(self.left_bound_idx, self.right_bound_idx)
        y_coord_grain = random.randint(self.top_bound_idx, self.bot_bound_idx)

        # print('y_coord_grain, x_coord_grain')
        # print(y_coord_grain, x_coord_grain)

        #increment the count at that location
        self.drop_sandgrain_at_pos(x_coord_grain, y_coord_grain)
        # print(self.grid)

    def avalanche(self,):
        # print('AVALANCHING')
        
        # for agent in self.agents:
            # print('agent_pos (i,j) (Y, X) before avalanche: ', agent.get_agent_pos())
            # self.print_grid_and_agent_pos(agent)

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

        # print('y_coord_unstable, x_coord_unstable')
        # print(y_coord_unstable, x_coord_unstable)


        # topple the grid at this coordinate
        self.grid[y_coord_unstable, x_coord_unstable] -= self.MAXIMUM_GRAINS

        # increment neighboring vertex counts
        self.increment_neighbors(y_coord_unstable, x_coord_unstable)

        # print('POST TOPPLE')
        # self.print_grid()

        # move the agent to one of the neighbors if the agent was at the unstable coordinate
        for agent in self.agents:
            # check if agent is at the coordinate
            if agent.is_in_game():
                agent_is_at_unstable_pos = agent.agent_is_at_pos(x_coord_unstable, y_coord_unstable)

            if agent.is_in_game() and agent_is_at_unstable_pos:
                
                # print('moving agent due to avalanche')
                agent.move_agent_random_from_point()

                # if the agent was avalanched, subtract a point
                agent.append_reward(-1)
                
                #update agent is_getting_avalanched
                agent.set_is_getting_avalanched(True)

                # print('agent pos after avalanche (i,j) (Y, X)', agent.get_agent_pos())
                # self.print_agent_pos_on_grid(agent)
                # input()

                # check if agent is still in game
                if not self.check_agent_is_in_grid(agent):
                    agent.remove_agent_from_game()
                    # print('REMOVING AGENT FROM GAME FROM AVALANCHE')
        # input()

        # update avalanching state
        self.is_avalanching = np.any(self.grid >= self.MAXIMUM_GRAINS)
        
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

def run_sandpile_alone(N_grid=2, initial_grid=None, MAXIMUM_GRAINS=4, DROP_SAND=True, MAX_STEPS=1000):
    # runs the sandpile for MAX_STEPS iterations and returns it
    sandpile = Sandpile(N_grid=N_grid, initial_grid=initial_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=DROP_SAND, MAX_STEPS=MAX_STEPS)

    for _ in range(MAX_STEPS):
        sandpile.step()

    return sandpile.get_sandpile()
    

        
