# Udacity Deep Reinforcement Learning Nanodegree Project 3: Collaboration and Competition

## Project description

![Trained Agent](https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif)

In this project two agents control rackets to bounce a ball over a net in the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.
If an agent hits the ball over the net, it receives a reward of +0.1.
If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  
Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. 
Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

## Setup

Setup the dependencies as described [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/README.md).

Download the environment from one of the links below.
You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
For Windows users, check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

If you'd like to train the agent on AWS and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), 
then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment. 
You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. 
To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.

Clone the repository and unpack the environment file in the project folder.


### Solving the Environment

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.




## Instructions

- To run the project just execute the <b>Tennis_Control.py</b> file.
- There is also an Tennis_Control.ipynb file for jupyter notebook execution.
- The <b>MultiEnvCheckPt/*checkpoint.pth</b> has the checkpointed actor and critic models
- environment.yml has the list of dependencies. The list includes few packages which are not used.

### Code Description

1. Tennis_Control.ipynb - Main module containing 1)loading of the helper modules 2) loading of the DDPG/MADDPG agent(s) helper module 3)training the  agent(s) 4)plotting results and 5) checkpointing the model parameters.
2. model.py - loads pytorch module and derives a custom NN model for this problem
3. ddpg_agent.py - Helper module contains 1) loads the helper model.py module 2)uses the NN model to train a DDPG agent 3) Contains experience replay buffer from which the DQN draws sample 
4. maddpg_agent.py - Helper module contains 1) loads the helper model.py module 2)uses the NN model to train a MADDPG agent 3) Contains experience replay buffer from which the DQN draws sample 
5. ddpg_agent_utils.py - Helper classes and methods used by both the ddpg_agent.py and maddpg_agent.py modules


### Important hyperparameters:
1. Continuous_Control.ipynb - Main Module contains most of the hyper parameters for training the DDPG agent 
  - n_episodes Size: 1500     # Maximum number of episodes for which training will proceed
  - checkpoint_score: 1     # if the score is greater than this threshold, network is checkpointed and training is finished. 



2. ddpg_agent.py - contains most of the agent parameters
  - Learning Rate: 1e-3/2e-3 ( DNN actor/critic) # learning rate 
  - Batch Size: 128     # minibatch size
  - Replay Buffer: 1e6  # replay buffer size
  - Gamma: 0.99         # discount factor
  - Tau: 3e-3           # for soft update of target parameters
  - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.) # Noise use to introduce entropy in the system to explore more

3. model.py contains the NN architecture and associated parameters
- For the neural models:    
  - Actor    
    - Hidden: (input, 256)  - ReLU
    - Hidden                - BatchNorm
    - Hidden: (256, 128)    - ReLU
    - Hidden                - BatchNorm
    - Output: (128, 4)      - TanH

  - Critic (for DDPG)
    - Hidden: (input, 128)                - ReLU
    - Hidden                              - BatchNorm
    - Hidden: (128 + action_size=2, 128)  - ReLU
    - Hidden                              - BatchNorm
    - Output: (128, 1)                  ' - Linear

  - Critic (for MADDPG)
    - Hidden: (input x nume of agent(=2) , 128)                     - ReLU
    - Hidden                                                        - BatchNorm
    - Hidden: (128 + nume of agent(=2) x action_size(=2)=4, 128)    - ReLU
    - Hidden                                                        - BatchNorm
    - Output: (128, 1)  