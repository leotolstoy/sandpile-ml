# Reinforcement Learning to Solve the Abelian Sandpile Game

## Introduction

This repository contains a PyTorch implementation of a Reinforcement Learning (RL) policy agent using the REINFORCE/policy gradient algorithm. The agent is trained to maximize the score in a simulated Abelian sandpile game, which is a classic system exhibiting self-organized criticality, and is meant to provide a brief investigation to neural networks' ability to learn environments exhibiting self-organized criticality.

### Abelian Sandpile Game

The [Abelian sandpile model](https://en.wikipedia.org/wiki/Abelian_sandpile_model) is a cellular automaton that exhibits self-organized criticality. In this game, grains of sand are dropped onto a grid, and when a cell exceeds a certain threshold, it topples and distributes sand to its neighboring cells. The dynamics of the sandpile lead to emergent patterns and critical behavior, in which large 'avalanches' (large series of grains falling and sequentially displacing neighbors) cannot be easily anticipated. As with other systems exhibiting self-organized criticality, the dynamics of the sandpile game exhibit scale invariance (the size of the avalanche follows a power law distribution, where larger avalanches are less likely).

### Policy Gradients

The RL agent in this repository utilizes the *REINFORCE algorithm*, a policy gradient method. Policy gradients directly optimize the policy of the agent.

## Usage
### Dependencies
- PyTorch
- NumPy
- Matplotlib

### Training

To train the RL policy agent from scratch, use the following script:

```bash
python train_reinforce_agent.py
```


### Testing
To simulate the RL policy agent on the Abelian sandpile game for a single run, run the following notebook:
```bash
sim_rl_agent.ipynb
```

To simulate the agent on multiple runs to calculate statistics (e.g. percentiles of scores)
```bash
sim_rl_agent_multiple_runs.ipynb
```


### Animation
To generate an animation of the RL agent interacting with the sandpile for a single run, use the following script:
```bash
python animate_rl_agent.py
```

## How It Works
### Agent Functionality
[agents.py](agents.py): Implements the generic agent functionality in the simulation, including interaction with the sandpile environment. 

### RL Policy Class Implementations
[rl_agents.py](rl_agents.py): Contains the PyTorch implementation of the RL policy class.
### Tests
The [tests.py](tests.py) folder contains unit tests for different components of the codebase, such as testing sandpile mechanics or agent movement.


## Contact

If you have any questions or would like to reach out to me, you can find me here: <br />
Email: leomed07@gmail.com <br />
