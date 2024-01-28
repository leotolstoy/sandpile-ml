import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from time import time, sleep
import random
from util import Directions
from sandpile import Sandpile, run_sandpile_alone
from agents import RandomAgent, MaxAgent, SeekSpecificValueAgent, SeekCenterAgent
import torch
from torch_util import enum_parameters
from rl_agents import Policy

N_grid = 10 #number of cells per side
# N_tick_step = 5
N_tick_step = 1

MAXIMUM_GRAINS = 4
N_runs = 10

I = 0
fig = plt.figure()

#these bounds show the sandpile
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

DO_EXPORT_ANIM = True
AGENT_COLOR_CODES = ['r', 'b', 'k']


# Run the best model
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


# SET UP POLICY AGENT
N_grid = 10
num_hidden_layers = 4
hidden_dim = 64
input_dim = ((2*N_grid-1)**2) # The number of input variables. 
output_dim = len(Directions) # The number of output variables. 

rl_policy_agent = Policy(
    input_dim=input_dim,
    num_hidden_layers=num_hidden_layers,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    device=device
)
enum_parameters(rl_policy_agent)
rl_policy_agent.to(device)

model_nickname = 'reinforce-agent'

# model_dir = f'/staging_area/{model_nickname}/'
model_dir = ''

checkpoint = torch.load(model_dir+'best_rl_policy_agent.tar')
g = checkpoint['model_state_dict']
score = checkpoint['score']
print(f'Best Score: {score}')
rl_policy_agent.load_state_dict(g)

# Put model in evaluation mode
rl_policy_agent.eval()

# start new sandpile with initial grid
rl_policy_agent.reset()


# initialize regular agents with random positions
random_agent = RandomAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))
max_agent = MaxAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))
ssv_agent = SeekSpecificValueAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1),specific_value=1)
center_agent = SeekCenterAgent(x_pos_init=random.randint(0,N_grid-1), y_pos_init=random.randint(0,N_grid-1))


# aggregate agents
agents = [rl_policy_agent, random_agent, center_agent]


# generate initial grid
# run the sandpile 1000 times
initial_grid_N = 1000
print('Generating initial grid')
initial_grid = run_sandpile_alone(N_grid=N_grid, initial_grid=None, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=True, MAX_STEPS=initial_grid_N)
print('initial grid')
print(initial_grid)
sandpile = Sandpile(N_grid=N_grid, initial_grid=initial_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=agents, MAX_STEPS=N_runs, STORE_STATE_BUFFER=True)


fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs)

# run the simulation
for i in range(N_runs):
    print(i)
    
    sandpile.step()

# M = len(grid_buffer)
grid_buffer = sandpile.grid_buffer # M x (N_grid x N_grid)

# P = number of agents
# positions in (i,j) convention
agent_positions = sandpile.agent_positions # M x (P x 2)
# print(agent_positions)
print('grid_buffer', len(grid_buffer))
print('agent_positions', len(agent_positions))

# input()

# loop through the grid buffer
frames = len(grid_buffer)

def init():
    """initialize animation"""
    img = axs.imshow(grid_buffer[0],cmap=cmap,norm=norm, origin="lower")
    agent_positions_step = agent_positions[0]

    for i, pos in enumerate(agent_positions_step):
        pos_i = pos[1]
        pos_j = pos[0]
        axs.plot(pos_i, pos_j, color=AGENT_COLOR_CODES[i], marker='o', markersize=36)

    return axs,

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds


def animate(i):
    # print(i)
    axs.cla()  
    img = axs.imshow(grid_buffer[i],cmap=cmap,norm=norm, origin="lower")

    agent_positions_step = agent_positions[i]

    for kk, pos in enumerate(agent_positions_step):
        pos_i = pos[1]
        pos_j = pos[0]
        # print(pos)
        axs.plot(pos_i, pos_j, color=AGENT_COLOR_CODES[kk], marker='.', markersize=36)

    return axs, 

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, repeat=False, init_func=init)
anim.save('animation.gif', writer='imagemagick', fps=2)

#print final grid
sandpile.print_grid()

# get avalance sizes
avalanche_sizes = np.array(sandpile.avalanche_sizes)

#plot histogram and loglog
bins = 10
hist_vals, x_recon = np.histogram(avalanche_sizes, bins=bins, density=True)

fig_hist, axs_hist = plt.subplots(2,1)
axs_hist[0].hist(avalanche_sizes,bins=bins, density=True)
axs_hist[1].loglog(x_recon[:-1],hist_vals,color='r',marker='o')

# img = axs.imshow(grid)


#manually log avalanche sizes and then plot histogram
log_avalanche_sizes = np.log10(avalanche_sizes)

bins = 10
hist_vals, x_recon = np.histogram(log_avalanche_sizes, bins=bins, density=True)

fig_hist, axs_hist = plt.subplots(2,1)
axs_hist[0].hist(log_avalanche_sizes,bins=bins, density=True)
axs_hist[1].plot(np.log10(x_recon[:-1]),hist_vals,color='r',marker='o')
plt.show()

