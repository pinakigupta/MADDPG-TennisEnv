#!/usr/bin/env python
# coding: utf-8

# # Continuous Control
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!

# The environments corresponding to both versions of the environment are already saved in the Workspace and can be accessed at the file paths provided below.  
# 
# Please select one of the two options below for loading the environment.

# In[3]:


from unityagents import UnityEnvironment
import numpy as np

# select this option to load version 1 (with a single agent) of the environment

env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64', no_graphics=True)


# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

# In[4]:


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
for brains in env.brain_names:
    print(brains)


# ### 2. Examine the State and Action Spaces
# 
# Run the code cell below to print some information about the environment.

# In[5]:


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_envs = len(env_info.agents)
print('Number of agents:', num_envs)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# In[31]:


type(states[0])


# ### 3. Loading DDPG Agent
# 

# In[6]:


from importlib import reload 
from collections import deque
import ddpg_agent, maddpg_agent
reload(ddpg_agent)
reload(maddpg_agent)
import torch
import torch.nn.functional as F
import torch.optim as optim
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE is ", DEVICE)
from ddpg_utils import OUNoise, Replay, transpose_to_tensor, Config
import model
reload(model)

# In[25]:


import matplotlib.pyplot as plt

def plot_scores(scores,avg_scores = None):
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y_label = 'Score'
    plt.plot(np.arange(len(scores)), scores)
    if avg_scores is not None:
        y_label+=', Avg Score'
        plt.plot(np.arange(len(scores)), avg_scores)
    plt.ylabel(y_label)
    plt.xlabel('Episode #')
    plt.show()


# In[18]:


def train(env = None, n_episodes=1000, agent = None, 
         checkpoint_score = 0.5, checkpt_folder = ""):
    
    
    scores_deque = deque(maxlen=100)
    scores = []
    goal_steps = []
    mean_scores_window = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations            # get the current state
        agent.reset()
        score = np.zeros(num_envs)
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states = env_info.vector_observations      # get the next states
            rewards = env_info.rewards                      # get the reward
            dones = env_info.local_done                     # see if episode has finished
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if np.any(dones):
                break

        score = score.max()
        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        mean_scores_window.append(np.mean(scores_deque))

        print('\rEpisode {}\tAverage Score: {:.2f}\tscore: {:.2f}'.
              format(i_episode, np.mean(scores_deque), scores[-1]), end="")


        if np.mean(scores_deque)>=checkpoint_score:
            checkpt = "Episode" + str(i_episode)
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            agent.checkpoint(checkpt)
            break   
    
    return scores, mean_scores_window


# In[19]:

config = Config(seed=6)

config.num_agents = len(env_info.agents)
config.state_size = state_size
config.action_size = action_size

config.actor_fn = lambda: model.Actor(config.state_size, config.action_size, 128, 128)
config.actor_opt_fn = lambda params: optim.Adam(params, lr=1e-3)

config.critic_fn = lambda: model.Critic(config.state_size, config.action_size , config.num_agents, 128, 128)
config.critic_opt_fn = lambda params: optim.Adam(params, lr=2e-3)

config.replay_fn = lambda: Replay(config.action_size, buffer_size=int(1e6), batch_size=128)
config.noise_fn = lambda: OUNoise(config.action_size, mu=0., theta=0.15, sigma=0.1 , seed=config.seed )

config.discount = 0.99
config.target_mix = 3e-3

config.max_episodes = 2000
config.max_steps = int(1e6)
config.goal_score = 1

config.CHECKPOINT_FOLDER = "MultiAgentCheckPt"


agent = maddpg_agent.Agent(config=config)


# In[21]:


ddpg_scores , ddpg_avg_scores  = train( env = env,
                                        agent = agent,
                                        n_episodes = 3000) # Multiple parallel Env


# In[26]:


plot_scores(ddpg_scores, ddpg_avg_scores) # random replay scores


# When finished, you can close the environment.

# In[6]:


env.close()

