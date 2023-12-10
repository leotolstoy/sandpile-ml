import numpy as np
import random
from matplotlib import pyplot as plt
from time import time

from util import Directions
from sandpile import Sandpile
from agents import RandomAgent, MaxAgent, SeekSpecificValueAgent

N_grid = 5 #number of cells per side

MAXIMUM_GRAINS = 4
max_nmoves = 10000
N_RUNS = 100

X_POS_INIT = N_grid//2
Y_POS_INIT = N_grid//2

random_agent_cumulative_scores = []
max_agent_cumulative_scores = []
ssv_agent_cumulative_scores = []

random_agent_nmoves = []
max_agent_nmoves = []
ssv_agent_nmoves = []



for _ in range(N_RUNS):
    # initialize agents
    random_agent = RandomAgent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT)
    max_agent = MaxAgent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT)
    ssv_agent = SeekSpecificValueAgent(x_pos_init=X_POS_INIT, y_pos_init=Y_POS_INIT,specific_value=1)
    agents = [random_agent, max_agent, ssv_agent]

    sandpile = Sandpile(N_grid=N_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=agents, MAX_STEPS=max_nmoves)
    REWARD_OFF_GRID = sandpile.REWARD_OFF_GRID

    i = 0
    game_is_running = sandpile.step()
    while game_is_running:
        # print(i)
        i+=1
        game_is_running = sandpile.step()


    # aggregate scores and moves
    random_agent_cumulative_score = random_agent.cumulative_score - REWARD_OFF_GRID
    max_agent_cumulative_score = max_agent.cumulative_score - REWARD_OFF_GRID
    ssv_agent_cumulative_score = ssv_agent.cumulative_score - REWARD_OFF_GRID

    random_agent_nmoves_run = len(random_agent.moves)
    max_agent_nmoves_run = len(max_agent.moves)
    ssv_agent_nmoves_run = len(ssv_agent.moves)
    
    random_agent_cumulative_scores.append(random_agent_cumulative_score)
    max_agent_cumulative_scores.append(max_agent_cumulative_score)
    ssv_agent_cumulative_scores.append(ssv_agent_cumulative_score)

    random_agent_nmoves.append(random_agent_nmoves_run)
    max_agent_nmoves.append(max_agent_nmoves_run)
    ssv_agent_nmoves.append(ssv_agent_nmoves_run)


    # print('cumulative random_agent score: ', random_agent_cumulative_score)
    # print('cumulative max_agent score: ', max_agent_cumulative_score)
    # print('cumulative ssv_agent score: ', ssv_agent_cumulative_score)

    # print('N_moves random_agent score: ', random_agent_nmoves)
    # print('N_moves max_agent score: ', max_agent_nmoves)
    # print('N_moves ssv_agent score: ', ssv_agent_nmoves)
    # input()

# Compute statistics on quantiles
# The distributions of nmoves and cumulative rewards are not Gaussian, 
# so just take the quantiles at 0.01, 0.5, and 0.99 to parameterize them

random_agent_cumulative_scores = np.array(random_agent_cumulative_scores)
max_agent_cumulative_scores = np.array(max_agent_cumulative_scores)
ssv_agent_cumulative_scores = np.array(ssv_agent_cumulative_scores)
random_agent_nmoves = np.array(random_agent_nmoves)
max_agent_nmoves = np.array(max_agent_nmoves)
ssv_agent_nmoves = np.array(ssv_agent_nmoves)

quantiles = [0.01, 0.5, 0.99]
random_agent_cumulative_scores_quantiles = np.quantile(random_agent_cumulative_scores, quantiles)
max_agent_cumulative_scores_quantiles = np.quantile(max_agent_cumulative_scores, quantiles)
ssv_agent_cumulative_scores_quantiles = np.quantile(ssv_agent_cumulative_scores, quantiles)

random_agent_nmoves_quantiles = np.quantile(random_agent_nmoves, quantiles)
max_agent_nmoves_quantiles = np.quantile(max_agent_nmoves, quantiles)
ssv_agent_nmoves_quantiles = np.quantile(ssv_agent_nmoves, quantiles)

print('Random Agent')
print('Cumulative Score Quantiles: ', random_agent_cumulative_scores_quantiles)
print('N Moves Quantiles: ', random_agent_nmoves_quantiles)

print()
print('Max Agent')
print('Cumulative Score Quantiles: ', max_agent_cumulative_scores_quantiles)
print('N Moves Quantiles: ', max_agent_nmoves_quantiles)

print()
print('SSV Agent')
print('Cumulative Score Quantiles: ', ssv_agent_cumulative_scores_quantiles)
print('N Moves Quantiles: ', ssv_agent_nmoves_quantiles)


n_bins = 20
fig_rewards, axs_rewards = plt.subplots(2,1)
axs_rewards[0].hist(random_agent_cumulative_scores,color='r',label='Random Agent',alpha=0.3,bins=n_bins)
axs_rewards[0].hist(max_agent_cumulative_scores,color='b',label='Max Agent',alpha=0.3,bins=n_bins)
axs_rewards[0].hist(ssv_agent_cumulative_scores,color='g',label='SSV Agent',alpha=0.3,bins=n_bins)

axs_rewards[0].legend()
axs_rewards[0].set_xlabel('Cumulative Rewards')

axs_rewards[1].hist(random_agent_nmoves,color='r',label='Random Agent',alpha=0.3,bins=n_bins)
axs_rewards[1].hist(max_agent_nmoves,color='b',label='Max Agent',alpha=0.3,bins=n_bins)
axs_rewards[1].hist(ssv_agent_nmoves,color='g',label='SSV Agent',alpha=0.3,bins=n_bins)


axs_rewards[1].set_xlabel('Number of Moves')

plt.show()


