# import tensorflow as tf
# tf.config.list_physical_devices('GPU')
# tf.test.is_built_with_cuda()
import os, sys
sys.path.append('../')
import torch
import torch.nn as nn 
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from save_best_model import SaveBestModel
from sandpile import Sandpile, run_sandpile_alone
import random
from collections import deque
from torch.distributions import Categorical
import time
import datetime
from rl_agents import Policy
from util import Directions
from torch_util import enum_parameters

"""This script trains an RL agent on the sandpile
"""

# Set the seed value all over the place to make this reproducible.
seed_val = 42


random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
model_nickname = 'reinforce-agent'

output_dir = f'/staging_area/{model_nickname}/'

# # Create output directory if needed
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

best_model_name = 'best_rl_policy_agent.tar'
# save_best_model = SaveBestModel(output_dir+best_model_name)
save_best_model = SaveBestModel(best_model_name)

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")


if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(rl_policy)


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

MAXIMUM_GRAINS = 4
max_nmoves_per_episode = 1000

agents = [rl_policy_agent]
sandpile = Sandpile(N_grid=N_grid, initial_grid=None, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=None, MAX_STEPS=10)
# rl_policy_agent.select_action(sandpile, 0, 0)
# rl_policy_agent.select_action(sandpile, 0, N_grid-1)
rl_policy_agent.select_action(sandpile, 4, 4)
# rl_policy_agent.select_action(sandpile, N_grid-1, N_grid-1)

N_training_episodes = 10000
N_val_episodes = 100
N_val_frequency = 500
N_print = 100

gamma = 0.999
beta_entropy = 1e1


training_scores = []
validation_scores = []
# Measure the total training time for the whole run.
total_t0 = time.time()

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

agents = [rl_policy_agent]

optimizer = torch.optim.Adam(rl_policy_agent.parameters(), lr=1e-4, betas=(0.9, 0.998), eps=1e-9, weight_decay=1e-4)

print("")
print('Training...')
# Measure how long the total training epoch takes.
t0 = time.time()

for i_episode in range(1, N_training_episodes+1):
    rl_policy_agent.train()
    # ========================================
    #               Training
    # ========================================
    t_episode_start = time.time()
    # print('i_episode: ', i_episode)
    
    # generate initial grid
    initial_grid_N = N_grid * N_grid * 4
    # print('Generating initial grid')
    initial_grid = run_sandpile_alone(N_grid=N_grid, initial_grid=None, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=True, MAX_STEPS=initial_grid_N)
    # print(initial_grid)

    # start new sandpile with initial grid
    rl_policy_agent.reset()
    sandpile = Sandpile(N_grid=N_grid, initial_grid=initial_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=agents, MAX_STEPS=max_nmoves_per_episode, grain_loc_order=None)
    
    # move agent to random position at beginning of episode
    rl_policy_agent.move_agent_to_point(random.randint(0,N_grid-1), random.randint(0,N_grid-1))
    pos = rl_policy_agent.get_agent_pos()

    episode_rewards = []
    agent_moves = []
    log_probs = []
    entropies = []
    i = 0
    game_is_running = True
    while game_is_running:
        # print('Step i: ', i)
        i+=1
        sandpile_grid, agent_rewards, game_is_running = sandpile.step()
        pos = rl_policy_agent.get_agent_pos()

        # get action and log prob
        action = rl_policy_agent.action_idx
        log_prob = rl_policy_agent.log_prob
        entropy = rl_policy_agent.entropy

        #only one agent is running so agent_rewards is a list with one element
        reward = agent_rewards[0]

        log_probs.append(log_prob)
        entropies.append(entropy)

        episode_rewards.append(reward)
        agent_moves.append(list(Directions)[action])

        # input()

    
    cumulative_score_episode = np.sum(episode_rewards)
    training_scores.append(cumulative_score_episode)

    returns = deque(maxlen=max_nmoves_per_episode)
    n_steps_episode = len(episode_rewards)

    for t in range(n_steps_episode)[::-1]:
        discounted_return_t = returns[0] if len(returns) > 0 else 0
        returns.appendleft(gamma * discounted_return_t + episode_rewards[t])

    eps = np.finfo(np.float32).eps.item()
    returns = torch.tensor(returns)

    policy_loss = 0


    for log_prob, disc_return in zip(log_probs, returns):
        # print('log_prob ', log_prob)
        # print('disc_return ', disc_return)

        policy_loss += (-log_prob * disc_return)
        
    entropies = torch.tensor(entropies)
    entropy_loss = torch.sum(entropies)

    loss = policy_loss - (beta_entropy * entropy_loss)
    optimizer.zero_grad()   
    loss.backward()
    optimizer.step()


    # Progress update every 100 batches.
    if i_episode % N_print == 0 and not i_episode == 0:
        # Calculate elapsed time in minutes.
        elapsed = format_time(time.time() - t0)
        
        # Report progress.
        print('  Episode {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(i_episode, N_training_episodes, elapsed))
        print('episode_rewards: ', episode_rewards)
        print('agent_moves: ', agent_moves)

        # print('episode_rewards', episode_rewards)
        print('cumulative_score_episode', cumulative_score_episode)
        print('n_steps_episode', n_steps_episode)

        # print('log_probs: ', log_probs)
        # print('returns: ', returns)
        # print('entropies: ', entropies)
        # policy_loss = torch.tensor(policy_loss, requires_grad=True).sum()
        print('policy_loss: ', policy_loss.item())
        # print('policy_loss grad ', policy_loss.grad)
        print('entropy_loss: ', entropy_loss.item())
        print()

    # periodically evaluate model and save it
    if i_episode % N_val_frequency == 0 and i_episode != 0:

        print("")
        print("Running Validation...")

        # ========================================
        #               Validation
        # ========================================

        t0_val = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        rl_policy_agent.eval()

        total_val_score = 0
        with torch.no_grad():
            for val_episode_i in range(N_val_episodes):

                # generate initial grid
                # run the sandpile 1000 times
                initial_grid_N = N_grid * N_grid * 4
                # print('Generating initial grid')
                initial_grid = run_sandpile_alone(N_grid=N_grid, initial_grid=None, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=True, MAX_STEPS=initial_grid_N)


                # start new sandpile with initial grid
                rl_policy_agent.reset()
                sandpile = Sandpile(N_grid=N_grid, initial_grid=initial_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=agents, MAX_STEPS=max_nmoves_per_episode, grain_loc_order=None)

                # move agent to random position at beginning of episode
                rl_policy_agent.move_agent_to_point(random.randint(0,N_grid-1), random.randint(0,N_grid-1))

                i = 0
                game_is_running = True
                episode_rewards = []
                while game_is_running:
                    # print(i)
                    i+=1
                    sandpile_grid, agent_rewards, game_is_running = sandpile.step()

                    # get action and log prob
                    action = rl_policy_agent.action_idx
                    log_prob = rl_policy_agent.log_prob

                    #only one agent is running so agent_rewards is a list with one element
                    reward = agent_rewards[0]
                    log_probs.append(log_prob)

                    episode_rewards.append(reward)

                cumulative_score_episode = np.sum(episode_rewards)
                
                total_val_score += cumulative_score_episode

        total_val_score = total_val_score/N_val_episodes

        print('total_val_score: ', total_val_score)
        validation_scores.append(total_val_score)

        #save best model
        save_best_model(
            total_val_score, i_episode, rl_policy_agent, optimizer
        )

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0_val)

        # print("  Validation Score: {0:.4f}".format(avg_val_score))
        print("  Validation took: {:}".format(validation_time))

# Measure how long this episode took.
training_time = format_time(time.time() - t0)
print("  Training took: {:}".format(training_time))
print()


print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

print("Saving model to %s" % output_dir)

#save model params
# model_name = 'rl_policy_params.pt'
# torch.save(rl_policy.state_dict(), output_dir+model_name)

# model_name = 'rl_policy_full.pt'
# torch.save(rl_policy, output_dir+model_name)

#save model params
model_name = 'rl_policy_params.pt'
torch.save(rl_policy_agent.state_dict(), model_name)

model_name = 'rl_policy_full.pt'
torch.save(rl_policy_agent, model_name)


training_scores = np.array(training_scores)
validation_scores = np.array(validation_scores)
fig, axs = plt.subplots()
axs.plot(training_scores,'-',label='Train')
axs.set_ylabel('Scores')
axs.plot(validation_scores,'-',label='Val')
axs.set_xlabel('Episode')
axs.legend()
print(np.min(validation_scores))


# Evaluate the best model
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

best_rl_policy_agent = Policy(
    input_dim=input_dim,
    num_hidden_layers=num_hidden_layers,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    device=device
)
enum_parameters(best_rl_policy_agent)
best_rl_policy_agent.to(device)

# model_dir = f'../staging_area/{reinforce-agent}/'
model_dir = ''

checkpoint = torch.load(model_dir+'best_rl_policy_agent.tar')
g = checkpoint['model_state_dict']
score = checkpoint['score']
print(f'Best Score: {score}')
best_rl_policy_agent.load_state_dict(g)

# Put model in evaluation mode
best_rl_policy_agent.eval()

N_RUNS_TEST = 1000

t0 = time.time()
test_scores = []
agent_nmoves = []

for _ in range(N_RUNS_TEST):
    # start new sandpile with initial grid
    rl_policy_agent.reset()

    # # generate initial grid
    initial_grid_N = N_grid * N_grid * 4
    # print('Generating initial grid')
    initial_grid = run_sandpile_alone(N_grid=N_grid, initial_grid=None, MAXIMUM_GRAINS=MAXIMUM_GRAINS, DROP_SAND=True, MAX_STEPS=initial_grid_N)

    sandpile = Sandpile(N_grid=N_grid, initial_grid=initial_grid, MAXIMUM_GRAINS=MAXIMUM_GRAINS, agents=agents, MAX_STEPS=max_nmoves_per_episode, grain_loc_order=None)

    # move agent to random position at beginning of episode
    rl_policy_agent.move_agent_to_point(random.randint(0,N_grid-1), random.randint(0,N_grid-1))

    pos = rl_policy_agent.get_agent_pos()
    
    episode_rewards = []
    agent_moves = []
    i = 0
    game_is_running = True
    with torch.no_grad():
        while game_is_running:
            # print('Step i: ', i)
            i+=1
            sandpile_grid, agent_rewards, game_is_running = sandpile.step()
            pos = rl_policy_agent.get_agent_pos()
            # print('Agent pos (ij): ', pos[0], pos[1])

            # get action and log prob
            action = rl_policy_agent.action_idx
            #only one agent is running so agent_rewards is a list with one element
            reward = agent_rewards[0]

            episode_rewards.append(reward)
            agent_moves.append(list(Directions)[action])
            

            # input()

    # print('episode_rewards: ', episode_rewards)
    # print('agent_moves: ', agent_moves)
    cumulative_score_episode = np.sum(episode_rewards)
    test_scores.append(cumulative_score_episode)
    agent_nmoves.append(len(episode_rewards))

t1 = time.time()
t_elapsed = t1 - t0
print('Time to run sim: ', t_elapsed)


test_scores = np.array(test_scores)
agent_nmoves = np.array(agent_nmoves)
quantiles = [0.01, 0.5, 0.99]

test_scores_quantiles = np.quantile(test_scores, quantiles)
agent_nmoves_quantiles = np.quantile(agent_nmoves, quantiles)

print('RL Agent')
print('Cumulative Score Quantiles: ', test_scores_quantiles)
print('N Moves Quantiles: ', agent_nmoves_quantiles)

n_bins = 20
fig_rewards, axs_rewards = plt.subplots(2,1)

axs_rewards[0].hist(test_scores_quantiles,color='r',label='RL Agent',alpha=0.3,bins=n_bins)

axs_rewards[0].legend()
axs_rewards[0].set_xlabel('Cumulative Rewards')

axs_rewards[1].hist(agent_nmoves_quantiles,color='r',label='RL Agent',alpha=0.3,bins=n_bins)


axs_rewards[1].set_xlabel('Number of Moves')

plt.show()
