# Project 2: (Unity) Reacher Enviornment

<img src="https://camo.githubusercontent.com/7ad5cdff66f7229c4e9822882b3c8e57960dca4e/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f766964656f2e756461636974792d646174612e636f6d2f746f706865722f323031382f4a756e652f35623165613737385f726561636865722f726561636865722e676966">

## Introduction
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributed Training

For this project, we will provide you with two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment,  your agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.    


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file. 

### Instructions

- To run the project just execute the <b>Continuous_Control.py</b> file.
- There is also an Continuous_Control.ipynb file for jupyter notebook execution.
- The <b>MultiEnvCheckPt/*checkpoint.pth</b> has the checkpointed actor and critic models
- environment.yml has the list of dependencies. The list includes few packages which are not used.




### Overview:
- The task solved here refers to a continuous control problem where the agent must be able to reach and go along with a moving ball controlling its arms.
- It's a continuous problem because the action has a continuous value and the agent must be able to provide this value instead of just chose the one with the biggest value (like in discrete tasks where it should just say which action it wants to execute).
- The reward of +0.1 is provided for each step that the agent's hand is in the goal location, in this case, the moving ball.
- The environment provides 2 versions, one with just 1 agent and another one with 20 agents working in parallel.
- For both versions the goal is to get an average score of +30 over 100 consecutive episodes (for the second version, the average score of all agents must be +30).
- For the checkpointed solution the second version of the environment has been used. This turns out to converge much faster due to the collective experience from multiple environments
- For the solution a Deep Deterministic Policy Gradients algorithm has been used.



### Code Description

1. Continuous_Control.ipynb - Main module containing 1)loading of the helper modules 2) loading of the DQN agent helper module 3)training the DQN agent 4)plotting results and 5) checkpointing the model parameters.
2. model.py - loads pytorch module and derives a custom NN model for this problem
3. ddpg_agent.py - Helper module contains 1) loads the helper model.py module 2)uses the NN model to train a DDPG agent 3) Contains experience replay buffer from which the DQN draws sample 4)Also processes multiple environment parallely 





### Important hyperparameters:
1. Continuous_Control.ipynb - Main Module contains most of the hyper parameters for training the DDPG agent 
  - n_episodes Size: 1000     # Maximum number of episodes for which training will proceed
  - checkpoint_score: 30     # if the score is greater than this threshold, network is checkpointed and training is finished. 



2. ddpg_agent.py - contains most of the agent parameters
  - Learning Rate: 1e-4 (in both DNN actor/critic) # learning rate 
  - Batch Size: 128     # minibatch size
  - Replay Buffer: 1e5  # replay buffer size
  - Gamma: 0.99         # discount factor
  - Tau: 1e-3           # for soft update of target parameters
  - Ornstein-Uhlenbeck noise parameters (0.15 theta and 0.2 sigma.) # Noise use to introduce entropy in the system to explore more

3. model.py contains the NN architecture and associated parameters
- For the neural models:    
  - Actor    
    - Hidden: (input, 256)  - ReLU
    - Hidden: (256, 128)    - ReLU
    - Output: (128, 4)      - TanH

  - Critic
    - Hidden: (input, 256)              - ReLU
    - Hidden: (256 + action_size, 128)  - ReLU
    - Hidden: (128, 128)  - ReLU
    - Output: (128, 1)                  - Linear
