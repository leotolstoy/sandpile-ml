import numpy as np
import random

class Sandpile():
    """This class implementes the sandpile environment and mechanics
    """

    def __init__(self, N_grid=2, initial_grid=None, MAXIMUM_GRAINS=4, agents=[], DROP_SAND=True, MAX_STEPS=1000, grain_loc_order=None):
        """

        Args:
            N_grid (int, optional): The number of cells along one dimension. Grid will be N_grid x N_grid. Defaults to 2.
            initial_grid (np array, 2D, optional): Option to prespecify an initial sandpile grid. Defaults to None.
            MAXIMUM_GRAINS (int, optional): Maximum number of sandgrains before avalanching occurs. Defaults to 4.
            agents (list, optional): A list of agents to simulate. Defaults to [].
            DROP_SAND (bool, optional): Whether to drop sandgrains. Defaults to True.
            MAX_STEPS (int, optional): The maximum number of steps to simulate. Defaults to 1000.
            grain_loc_order (np array, Nx2, optional): Prespecify the location of sandgrains to drop to. Defaults to None.
        """

        # allow for initial grid configuration
        if initial_grid is not None:
            self.grid = initial_grid.copy()
            assert(initial_grid.shape[0]==N_grid and initial_grid.shape[1]==N_grid)
        else:
            self.grid = np.zeros((N_grid, N_grid))

        self.N_grid = N_grid

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
        self.grain_loc_order = grain_loc_order
        self.agents = agents
        self.MAX_STEPS = MAX_STEPS
        self.iteration = 0
        self.agent_rewards = []

    def step(self,):
        """Step the simulation mechanics once

        Returns:
            _type_: _description_
        """

        self.avalanche_size = 0
        
        # preallocate agent rewards
        self.agent_rewards_step = [0] * len(self.agents)

        # Choose an action for each of the agents, if any are present
        for i, agent in enumerate(self.agents):
            
            if agent.is_in_game():
                # print('agent_pos (i,j) (Y, X): ', agent.get_agent_pos())
                # print('Moving agent')

                # have the agent choose a direction to move in
                direction = agent.choose_move(self)
                # print('move: ', direction)
                
                agent.move_agent_in_direction(direction)
                agent.set_is_getting_avalanched(False)

            # check agent position to see if they're still in the grid
            if agent.is_in_game() and not self.check_agent_is_in_grid(agent):
                # print('AGENT WALKED OFF')
                agent.remove_agent_from_game()
                # self.agent_rewards_step[i] = np.min((-10, -agent.get_cumulative_score()))
                self.agent_rewards_step[i] = -10
                # self.agent_rewards_step[i] = -agent.get_cumulative_score()
                # agent.append_reward(-agent.get_cumulative_score())
                agent.append_reward(0)

        if self.DROP_SAND:
            self.drop_sandgrain()

        #update avalanchine state
        self.is_avalanching = np.any(self.grid >= self.MAXIMUM_GRAINS)

        # Avalanche 
        while self.is_avalanching:
            self.avalanche()
            self.avalanche_size += 1

        # update avalanche sizes
        if self.avalanche_size > 0:
            self.avalanche_sizes.append(self.avalanche_size)
            # print(self.avalanche_sizes)

        # update agent rewards and states
        for i, agent in enumerate(self.agents):
            if agent.is_in_game():
                
                # if the agent is not getting avalanched, reward the agent with the number of sandgrains
                # at its location
                if not agent.get_is_getting_avalanched():
                    y_pos, x_pos = agent.get_agent_pos()
                    grid_reward = self.grid[y_pos, x_pos]
                    # print('REWARD FOR MOVE: ', reward)
                    # agent.append_reward(reward)
                    self.agent_rewards_step[i] += grid_reward

                # if the agent is getting avalanched, no reward
                elif agent.get_is_getting_avalanched():
                    pass

                agent.append_reward(self.agent_rewards_step[i])
            else:
                # self.agent_rewards.append(-100)
                # self.agent_rewards_step[i] = -agent.get_cumulative_score()
                # self.agent_rewards_step[i] = np.min((-10, -agent.get_cumulative_score()))
                # self.agent_rewards_step[i] = -100
                pass

            

        self.iteration += 1

        # establish that at least one agent is in the game, or if no agents were initialized
        # set this to true so that we can keep looping
        at_least_one_agent_in_game = False or len(self.agents) == 0
        for agent in self.agents:
            at_least_one_agent_in_game = at_least_one_agent_in_game or agent.is_in_game()

        #check that the game is still running
        iterations_not_exceeded = self.iteration < self.MAX_STEPS
        game_is_running = iterations_not_exceeded and at_least_one_agent_in_game
        
        if not game_is_running:
            pass
        # input()

        return self.grid, self.agent_rewards_step, game_is_running
    
    
    def check_agent_is_in_grid(self, agent):
        """Check if the agent is still in the grid bounds
        """
        in_x_pos = agent.x_pos >= self.left_bound_idx and agent.x_pos <= self.right_bound_idx
        in_y_pos = agent.y_pos >= self.top_bound_idx and agent.y_pos <= self.bot_bound_idx

        return in_x_pos and in_y_pos


    def drop_sandgrain(self,):
        """Drop a sandgrain, either randomly or on a preset location
        """
        if self.grain_loc_order is not None:
            sandgrain_pos = self.grain_loc_order[self.iteration, :]
            # print(sandgrain_pos)
            self.drop_sandgrain_at_pos(sandgrain_pos[0], sandgrain_pos[1])
        else:
            self.drop_sandgrain_randomly()

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
        """Implements avalanche dynamics
        """
        # print('AVALANCHING')

        # find indices where avalanching/unstable
        # returns a Nx2 array of xy coordinates where N is the number of indices over the maximum
        avalanche_idxs = np.argwhere(self.grid >= self.MAXIMUM_GRAINS)
        N_avalanche_ixs = avalanche_idxs.shape[0]

        #pick an unstable vertex at random
        rand_idx = np.random.randint(N_avalanche_ixs)
        rand_unstable = avalanche_idxs[rand_idx,:]

        x_coord_unstable = rand_unstable[1]
        y_coord_unstable = rand_unstable[0]

        # topple the grid at this coordinate
        self.grid[y_coord_unstable, x_coord_unstable] -= self.MAXIMUM_GRAINS

        # increment neighboring vertex counts
        self.increment_neighbors(y_coord_unstable, x_coord_unstable)

        # move the agent to one of the neighbors if the agent was at the unstable coordinate
        for i, agent in enumerate(self.agents):
            # check if agent is at the coordinate
            if agent.is_in_game():
                agent_is_at_unstable_pos = agent.agent_is_at_pos(x_coord_unstable, y_coord_unstable)

            if agent.is_in_game() and agent_is_at_unstable_pos:
                
                # print('moving agent due to avalanche')
                agent.move_agent_random_from_point()

                # if the agent was avalanched, subtract a point
                self.agent_rewards_step[i] -= 1
                
                #update agent is_getting_avalanched
                agent.set_is_getting_avalanched(True)

                # check if agent is still in game
                if not self.check_agent_is_in_grid(agent):
                    agent.remove_agent_from_game()
                    # agent.append_reward(-agent.get_cumulative_score())
                    self.agent_rewards_step[i] = -100
                    agent.append_reward(0)
                    # print('REMOVING AGENT FROM GAME FROM AVALANCHE')

        # update avalanching state
        self.is_avalanching = np.any(self.grid >= self.MAXIMUM_GRAINS)

    def increment_neighbors(self, y_coord, x_coord):
        """Add one sandgrain to the neighbors (if they exist) of a 
        given location
        """
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
    """Runs the sandpile for MAX_STEPS iterations and returns it
    """
    sandpile = Sandpile(N_grid=N_grid, initial_grid=initial_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=DROP_SAND, MAX_STEPS=MAX_STEPS)

    for _ in range(MAX_STEPS):
        sandpile.step()

    return sandpile.get_sandpile()
    

        
