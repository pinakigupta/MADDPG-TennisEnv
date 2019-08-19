import numpy as np
import random
import copy
import os
from collections import namedtuple, deque
from importlib import reload 
from ddpg_utils import OUNoise, Replay, transpose_to_tensor
import model
import torch
import torch.nn.functional as F
import torch.optim as optim

reload(model)

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 3        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 3e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 2e-3        # learning rate of the critic
WEIGHT_DECAY = 0   	    # L2 weight decay

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed , num_agents = 1, checkpt_folder = "checkpt" , config = None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        self.config = config

        
        self.state_size = state_size
        self.num_agents = num_agents
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.CHECKPOINT_FOLDER = checkpt_folder

        # Actor Network (w/ Target Network)
        self.actor_actual = model.Actor(state_size, action_size, 128, 128).to(DEVICE)
        self.actor_target = model.Actor(state_size, action_size, 128, 128 ).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_actual.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_actual = model.Critic(state_size, action_size, num_agents, 128, 128).to(DEVICE)
        self.critic_target = model.Critic(state_size, action_size, num_agents, 128, 128).to(DEVICE)
        self.critic_optimizer = optim.Adam(self.critic_actual.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        '''if os.path.isfile(self.CHECKPOINT_FOLDER + 'checkpoint_actor.pth') and os.path.isfile(self.CHECKPOINT_FOLDER + 'checkpoint_critic.pth'):
            self.actor_actual.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + 'checkpoint_actor.pth'))
            self.actor_target.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + 'checkpoint_actor.pth'))

            self.critic_actual.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + 'checkpoint_critic.pth'))
            self.critic_target.load_state_dict(torch.load(self.CHECKPOINT_FOLDER + 'checkpoint_critic.pth'))'''

        # Noise process
        self.noise = OUNoise((num_agents, action_size), random_seed)

        # Replay memory
        self.memory = Replay(action_size, BUFFER_SIZE, BATCH_SIZE)

        Agent.hard_update(self.actor_target, self.actor_actual)
        Agent.hard_update(self.critic_target, self.critic_actual)
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        game_state = states.flatten()
        game_next_state = next_states.flatten()
        game_action = actions.flatten()    
        self.memory.add((game_state, game_action, rewards, game_next_state, dones))

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(DEVICE)
        self.actor_actual.eval()
        with torch.no_grad():
            action = self.actor_actual(state).cpu().data.numpy()
        self.actor_actual.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        game_states, game_actions, rewards, game_next_states, dones = transpose_to_tensor(experiences)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models

        game_actions_next = [self.actor_target(torch.split(game_next_states, self.state_size, dim=1)[i]) for i in range(self.num_agents)]

        game_actions_next = torch.cat(game_actions_next, dim=1)    
        Q_targets_next = self.critic_target(game_next_states.to(DEVICE), game_actions_next.to(DEVICE))
        # Compute Q targets for current states (y_i)
        game_rewards = rewards.sum(1, keepdim=True)
        game_done = dones.max(1, keepdim=True)[0]
        Q_targets = game_rewards + (gamma * Q_targets_next * (1 - game_done ))
        # Compute critic loss
        Q_expected = self.critic_actual(game_states, game_actions).to(DEVICE)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss


        game_actions_pred = [self.actor_target(torch.split(game_states, self.state_size, dim=1)[i]) for i in range(self.num_agents)]
        game_actions_pred = torch.cat(game_actions_pred, dim=1)
        actor_loss = -self.critic_actual(game_states.to(DEVICE), game_actions_pred.to(DEVICE)).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        Agent.soft_update(self.critic_target, self.critic_actual, TAU)
        Agent.soft_update(self.actor_target, self.actor_actual, TAU)                     

    @staticmethod
    def hard_update(target_model, source_model):
        for target_param, param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def soft_update(target_model, source_model, mix):
        for target_param, online_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - mix) + online_param.data * mix)

    def checkpoint(self,string):
        if not (os.path.isdir(self.CHECKPOINT_FOLDER)):
               os.makedirs(self.CHECKPOINT_FOLDER)
        torch.save(self.actor_actual.state_dict(), self.CHECKPOINT_FOLDER + '/'+string+'_actor.pth')      
        torch.save(self.critic_actual.state_dict(), self.CHECKPOINT_FOLDER + '/'+string+'_critic.pth')  

