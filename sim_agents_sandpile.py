import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl

from sandpile import Sandpile, run_sandpile_alone
from agents import RandomAgent, MaxAgent, SeekSpecificValueAgent

DO_ANIM = not True
N_grid = 10 #number of cells per side
# N_tick_step = 5
N_tick_step = 1

MAXIMUM_GRAINS = 4
N_runs = 10000
frames = N_runs
I = 0
fig = plt.figure()

#these bounds show the sandpile plus one square of void around it
# LIM_MIN = 1 - 1 - 0.5
# LIM_MAX = N_grid-0.5 + 1 + 1


LIM_MIN = 1 - 0.5
LIM_MAX = N_grid-0.5

axs = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(LIM_MIN, LIM_MAX), ylim=(LIM_MIN, LIM_MAX))
# axs.grid()
axs.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)

axs.set_xticks(np.arange(-.5, N_grid, N_tick_step))
axs.set_yticks(np.arange(-.5, N_grid, N_tick_step))



# axs.set_xticks(np.arange(-.5 + 1, N_grid+1, N_tick_step))
# axs.set_yticks(np.arange(-.5 + 1, N_grid+1, N_tick_step))
# axs.set_xticks([])
# axs.set_yticks([])

# axs.xaxis.set_tick_params(labelbottom=False)
# axs.yaxis.set_tick_params(labelleft=False)

# https://stackoverflow.com/questions/43971138/python-plotting-colored-grid-based-on-values
# https://stackoverflow.com/questions/7229971/2d-grid-data-visualization-in-python
# https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html
# cmap = plt.cm.viridis
cmap = plt.cm.get_cmap('Blues')
bounds = np.arange(0, MAXIMUM_GRAINS+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

X_POS_INIT = N_grid//2
Y_POS_INIT = N_grid//2

# initialize agents with random positions
random_agent = RandomAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))
max_agent = MaxAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))
ssv_agent = SeekSpecificValueAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1),specific_value=1)
agents = [random_agent, max_agent, ssv_agent]


# generate initial grid
# run the sandpile 1000 times
initial_grid_N = 1000
print('Generating initial grid')
initial_grid = run_sandpile_alone(N_grid=N_grid, initial_grid=None, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=True, MAX_STEPS=initial_grid_N)
print('initial grid')
print(initial_grid)


# start new sandpile with initial grid
sandpile = Sandpile(N_grid=N_grid, initial_grid=initial_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=agents, MAX_STEPS=N_runs)

# input()
def init():
    """initialize animation"""
    img = axs.imshow(sandpile.grid,cmap=cmap,norm=norm, origin="lower")
    fig.colorbar(img)
    
    return img,


def animate(i):
    # print(i)
    sandpile.step()
    img = axs.imshow(sandpile.grid)
    global I
    I += 1
    # print(I)
    return img, 

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds

if DO_ANIM:
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, repeat=False, init_func=init)

else:
    i = 0
    game_is_running = sandpile.step()
    while game_is_running:
        print(i)
        i+=1
        game_is_running = sandpile.step()

print('cumulative random_agent score: ', random_agent.cumulative_score)
print('cumulative max_agent score: ', max_agent.cumulative_score)
print('cumulative ssv_agent score: ', ssv_agent.cumulative_score)

fig_rewards, axs_rewards = plt.subplots(2,1)
axs_rewards[0].plot(random_agent.rewards,color='r',marker='o',label='Random Agent')
axs_rewards[0].plot(max_agent.rewards,color='b',marker='o',label='Max Agent')
axs_rewards[0].plot(ssv_agent.rewards,color='g',marker='o',label='SSV Agent')
axs_rewards[0].legend()
axs_rewards[0].set_ylabel('Rewards')
axs_rewards[0].set_ylim(bottom=-1, top = 5)

axs_rewards[1].plot(random_agent.cumulative_rewards,color='r',marker='o')
axs_rewards[1].plot(max_agent.cumulative_rewards,color='b',marker='o')
axs_rewards[1].plot(ssv_agent.cumulative_rewards,color='g',marker='o',label='SSV Agent')


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
plt.show()


