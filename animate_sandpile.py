import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from time import time, sleep
import random
from util import Directions
from sandpile import Sandpile, run_sandpile_alone
from agents import RandomAgent, MaxAgent, SeekSpecificValueAgent, SeekCenterAgent


N_grid = 10 #number of cells per side
# N_tick_step = 5
N_tick_step = 1

MAXIMUM_GRAINS = 4
N_runs = 500

fig = plt.figure()

#these bounds show the sandpile
LIM_MIN = 0 - 0.5
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

DO_AGENT_SIM = not True
DO_EXPORT_ANIM = True
AGENT_COLOR_CODES = ['r', 'b']

if DO_AGENT_SIM:
    # initialize agents with random positions
    random_agent = RandomAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))
    max_agent = MaxAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))
    ssv_agent = SeekSpecificValueAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1),specific_value=1)
    center_agent = SeekCenterAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))

    # agents = [random_agent, max_agent, ssv_agent, center_agent]
    agents = [random_agent, center_agent]

    # generate initial grid
    # run the sandpile 1000 times
    initial_grid_N = 1000
    print('Generating initial grid')
    initial_grid = run_sandpile_alone(N_grid=N_grid, initial_grid=None, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=True, MAX_STEPS=initial_grid_N)
    print('initial grid')
    print(initial_grid)
    sandpile = Sandpile(N_grid=N_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=agents, MAX_STEPS=N_runs, STORE_STATE_BUFFER=True)

else:
    sandpile = Sandpile(N_grid=N_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, MAX_STEPS=N_runs, STORE_STATE_BUFFER=True)

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs)
AGENT_NAMES = ['Random Agent', 'Center Agent']


# run the simulation
for i in range(N_runs):
    print(i)
    
    sandpile.step()

# M = len(grid_buffer)
grid_buffer = sandpile.grid_buffer # M x (N_grid x N_grid)

# P = number of agents
# positions in (i,j) convention
agent_positions = sandpile.all_agent_positions # M x (P x 2)
agent_rewards = sandpile.all_agent_rewards # N_runs x (P)
agent_iterations = sandpile.all_agent_iterations # M x (P)
agent_is_getting_avalanched = sandpile.all_agent_is_getting_avalanched # N_runs x (P)

# print(agent_positions)
print('grid_buffer', len(grid_buffer))
print('agent_positions', len(agent_positions))
print('agent_rewards', len(agent_rewards))
print('agent_iterations', len(agent_iterations)) # [0, 1, ..., N_runs-1], serves as index
print('agent_is_getting_avalanched', len(agent_is_getting_avalanched))
# input()

# loop through the grid buffer
frames = len(grid_buffer)

def animate(i):
    # print(i)
    # input()
    axs.cla()  
    img = axs.imshow(grid_buffer[i],cmap=cmap,norm=norm, origin="lower")

    agent_positions_step = agent_positions[i]

    for kk, pos in enumerate(agent_positions_step):
        pos_i = pos[1]
        pos_j = pos[0]
        # print(pos)
        axs.scatter(pos_i, pos_j, color=AGENT_COLOR_CODES[kk], marker='o', s=144, label=AGENT_NAMES[kk])
    axs.set_title(f'Step: {agent_iterations[i] + 1}/{N_runs}')
    axs.set_xlim(LIM_MIN, LIM_MAX)
    axs.set_ylim(LIM_MIN, LIM_MAX)
    # axs.legend(loc='lower center',ncol=2,frameon=False, bbox_to_anchor=(0.5, -0.05))
    return img, 

def init():
    return animate(0)

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, repeat=False, init_func=init)

if DO_EXPORT_ANIM:
    # FFwriter = animation.FFMpegWriter(fps=2)
    # anim.save('animation.mp4', writer=FFwriter)
    anim.save('raw_animation.gif', writer='imagemagick', fps=10)


plt.show()


