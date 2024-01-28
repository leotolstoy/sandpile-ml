import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl

from sandpile import Sandpile, run_sandpile_alone
from agents import RandomAgent, MaxAgent, SeekSpecificValueAgent, SeekCenterAgent

N_grid = 10 #number of cells per side

MAXIMUM_GRAINS = 4
N_runs = 1000


# initialize agents with random positions
random_agent = RandomAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))
max_agent = MaxAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))
ssv_agent = SeekSpecificValueAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1),specific_value=1)
center_agent = SeekCenterAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))

agents = [random_agent, max_agent, ssv_agent, center_agent]


# generate initial grid
# run the sandpile 1000 times
initial_grid_N = 1000
print('Generating initial grid')
initial_grid = run_sandpile_alone(N_grid=N_grid, initial_grid=None, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=True, MAX_STEPS=initial_grid_N)
print('initial grid')
print(initial_grid)


# start new sandpile with initial grid
sandpile = Sandpile(N_grid=N_grid, initial_grid=initial_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=agents, MAX_STEPS=N_runs)

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds

random_agent_pos = []
max_agent_pos = []
ssv_agent_pos = []
center_agent_pos = []

i = 0
game_is_running = True
while game_is_running:
    print(i)
    i+=1
    sandpile_grid, agent_rewards, game_is_running = sandpile.step()
    print(agent_rewards)

    random_agent_pos.append([random_agent.x_pos, random_agent.y_pos])
    max_agent_pos.append([max_agent.x_pos, max_agent.y_pos])
    ssv_agent_pos.append([ssv_agent.x_pos, ssv_agent.y_pos])
    center_agent_pos.append([center_agent.x_pos, center_agent.y_pos])


random_agent_pos = np.array(random_agent_pos)
max_agent_pos = np.array(max_agent_pos)
ssv_agent_pos = np.array(ssv_agent_pos)
center_agent_pos = np.array(center_agent_pos)



print('cumulative random_agent score: ', random_agent.cumulative_score)
print('cumulative max_agent score: ', max_agent.cumulative_score)
print('cumulative ssv_agent score: ', ssv_agent.cumulative_score)
print('cumulative center_agent score: ', center_agent.cumulative_score)

fig_rewards, axs_rewards = plt.subplots(2,1)
axs_rewards[0].plot(random_agent.rewards,color='r',marker='o',label='Random Agent')
axs_rewards[0].plot(max_agent.rewards,color='b',marker='o',label='Max Agent')
axs_rewards[0].plot(ssv_agent.rewards,color='g',marker='o',label='SSV Agent')
axs_rewards[0].plot(center_agent.rewards,color='k',marker='o',label='Center Agent')
axs_rewards[0].legend()
axs_rewards[0].set_ylabel('Rewards')
axs_rewards[0].set_ylim(bottom=-1, top = 5)

axs_rewards[1].plot(random_agent.cumulative_rewards,color='r',marker='o')
axs_rewards[1].plot(max_agent.cumulative_rewards,color='b',marker='o')
axs_rewards[1].plot(ssv_agent.cumulative_rewards,color='g',marker='o',label='SSV Agent')
axs_rewards[1].plot(center_agent.cumulative_rewards,color='k',marker='o',label='Center Agent')


axs_rewards[1].set_ylabel('Cumulative Rewards')
axs_rewards[1].set_ylim(bottom=-1)

# get avalance sizes
avalanche_sizes = np.array(sandpile.avalanche_sizes)

#plot histogram and loglog
bins = 10
hist_vals, x_recon = np.histogram(avalanche_sizes, bins=bins, density=True)

fig_hist, axs_hist = plt.subplots(2,1)
axs_hist[0].hist(avalanche_sizes,bins=bins, density=True)
axs_hist[1].loglog(x_recon[:-1],hist_vals,color='r',marker='o')

# img = axs.imshow(grid)

fig_pos, axs_pos = plt.subplots()
axs_pos.plot(random_agent_pos[:,0], random_agent_pos[:,1], color='r', marker='o',label='Random Agent')
axs_pos.plot(max_agent_pos[:,0], max_agent_pos[:,1], color='b', marker='o',label='Max Agent')
axs_pos.plot(ssv_agent_pos[:,0], ssv_agent_pos[:,1], color='g', marker='o',label='SSV Agent')
axs_pos.plot(center_agent_pos[:,0], center_agent_pos[:,1], color='k', marker='o',label='Center Agent')
axs_pos.legend()

plt.show()


