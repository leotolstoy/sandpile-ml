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
N_runs = 20

I = 0
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

DO_EXPORT_ANIM = True
# AGENT_COLOR_CODES = ['r', 'b']
AGENT_COLOR_CODES = ['r']


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
# agents = [random_agent, rl_policy_agent]
agents = [rl_policy_agent]


# generate initial grid
# run the sandpile 1000 times
initial_grid_N = 1000
print('Generating initial grid')
initial_grid = run_sandpile_alone(N_grid=N_grid, initial_grid=None, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=True, MAX_STEPS=initial_grid_N)
print('initial grid')
print(initial_grid)
sandpile = Sandpile(N_grid=N_grid, initial_grid=initial_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=agents, MAX_STEPS=N_runs, STORE_STATE_BUFFER=True)

# AGENT_NAMES = ['Random Agent', 'RL Agent']
AGENT_NAMES = ['RL Agent']

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs)

# run the simulation
for i in range(N_runs):
    print(i)
    
    sandpile.step()

grid_buffer = sandpile.grid_buffer # M x (N_grid x N_grid)
M = len(grid_buffer)

# P = number of agents
# positions in (i,j) convention
agent_positions = sandpile.all_agent_positions # M x (P x 2)
agent_rewards = sandpile.all_agent_rewards # N_runs x (P)
agent_rewards = np.array(agent_rewards)
agent_cumulative_rewards = np.cumsum(agent_rewards)
agent_iterations = sandpile.all_agent_iterations # M x (1)
agent_is_getting_avalanched = sandpile.all_agent_is_getting_avalanched # M x (P)
agent_moves = sandpile.all_agent_moves  # N_runs x (P)
is_avalanching_buffer = sandpile.is_avalanching_buffer # M x (1)

# print(agent_positions)
print('grid_buffer', len(grid_buffer))
print('agent_positions', len(agent_positions))
print('agent_rewards', len(agent_rewards))
print('agent_cumulative_rewards',len(agent_cumulative_rewards))
print('agent_iterations', len(agent_iterations)) # [0, 1, ..., N_runs-1], serves as index
print('agent_is_getting_avalanched', len(agent_is_getting_avalanched))
print(agent_is_getting_avalanched)
print('agent_moves', len(agent_moves))
print('is_avalanching_buffer', len(is_avalanching_buffer))
print(is_avalanching_buffer)
# input()

# loop through the grid buffer
frames = len(grid_buffer)

arrow_width = 0.1

def init():
    """initialize animation"""
    img = axs.imshow(grid_buffer[0],cmap=cmap,norm=norm, origin="lower")
    agent_positions_step = agent_positions[0]
    next_agent_positions_step = agent_positions[1]
    agent_is_getting_avalanched_step = agent_is_getting_avalanched[0]

    for kk, agent_pos_cur in enumerate(agent_positions_step):
        pos_i = agent_pos_cur[0] # y
        pos_j = agent_pos_cur[1] # x
        

        # compute motion to next step
        agent_pos_cur = agent_positions_step[kk]
        agent_pos_next = next_agent_positions_step[kk]

        dx = agent_pos_next[1] - agent_pos_cur[1]
        dy = agent_pos_next[0] - agent_pos_cur[0]
        # print('wdf')
        # print(agent_pos_cur)
        # print(agent_pos_next)

        # print(dx, dy)

        # draw agent arrow if the agent moved of its own volition and no avalanche is happening
        if not agent_is_getting_avalanched_step[kk] and not is_avalanching_buffer[0]:
            axs.arrow(pos_j, pos_i, dx, dy, width=arrow_width, color='k')
        elif agent_is_getting_avalanched_step[kk]:
            axs.arrow(pos_j, pos_i, dx, dy, width=arrow_width, color='r')

        # draw agent position
        axs.scatter(pos_j, pos_i, color=AGENT_COLOR_CODES[kk], marker='o', s=144, label=AGENT_NAMES[kk])
        # input()


    axs.set_title(f'Step: 1/{N_runs}, Agent Score: {agent_cumulative_rewards[0]}')

    return img,

# choose the interval based on dt and the time to animate one step
interval = 100 #delay between frames in milliseconds


def animate(i):
    # print(i)
    axs.cla()  
    img = axs.imshow(grid_buffer[i],cmap=cmap,norm=norm, origin="lower")

    agent_positions_step = agent_positions[i]
    agent_is_getting_avalanched_step = agent_is_getting_avalanched[i]

    if i < M-1:
        next_agent_positions_step = agent_positions[i+1]
    else:
        next_agent_positions_step = agent_positions[i]
            
    # draw agent positions
    for kk, pos in enumerate(agent_positions_step):
        pos_i = pos[0] # y pos
        pos_j = pos[1] # x pos
        # print(pos)
        

        # compute motion to next step
        agent_pos_cur = agent_positions_step[kk]
        agent_pos_next = next_agent_positions_step[kk]

        dx = agent_pos_next[1] - agent_pos_cur[1]
        dy = agent_pos_next[0] - agent_pos_cur[0]

        # print(agent_pos_cur)
        # print(agent_pos_next)

        # print(dx, dy)


        # draw agent arrow if the agent moved of its own volition and no avalanche is happening
        if not agent_is_getting_avalanched_step[kk] and not is_avalanching_buffer[i]:
            axs.arrow(pos_j, pos_i, dx, dy, width=arrow_width, color='k')
        elif agent_is_getting_avalanched_step[kk]:
            axs.arrow(pos_j, pos_i, dx, dy, width=arrow_width, color='r')

        # draw agent pos
        axs.scatter(pos_j, pos_i, color=AGENT_COLOR_CODES[kk], marker='o', s=144, label=AGENT_NAMES[kk])

    axs.set_xlim(LIM_MIN, LIM_MAX)
    axs.set_ylim(LIM_MIN, LIM_MAX)
    axs.set_title(f'Step: {agent_iterations[i] + 1}/{N_runs}, Agent Score: {agent_cumulative_rewards[agent_iterations[i]]}')
    
    return img, 

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=True, repeat=False, init_func=init)
if DO_EXPORT_ANIM:
    anim.save('raw_animation_rl_agent.gif', writer='imagemagick', fps=3)


plt.show()


