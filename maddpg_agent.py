import copy
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ddpg_utils import OUNoise, Replay, transpose_to_tensor
from importlib import reload 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Agent:
    
    def __init__(self, config):
        self.config = config

        self.actor_actual = config.actor_fn().to(DEVICE)
        self.actor_target = config.actor_fn().to(DEVICE)
        self.actor_optimizer = config.actor_opt_fn(self.actor_actual.parameters())

        self.critic_actual = config.critic_fn().to(DEVICE)
        self.critic_target = config.critic_fn().to(DEVICE)
        self.critic_optimizer = config.critic_opt_fn(self.critic_actual.parameters())

        self.noises = [config.noise_fn() for _ in range(config.num_agents)]
        self.replay = config.replay_fn()

        Agent.hard_update(self.actor_target, self.actor_actual)
        Agent.hard_update(self.critic_target, self.critic_actual)

    def act(self, states, add_noise=True):
        state = torch.from_numpy(states).float().to(DEVICE)
        self.actor_actual.eval()

        with torch.no_grad():
            action = self.actor_actual(state).cpu().numpy()

        self.actor_actual.train()
        if add_noise:
            action += [n.sample() for n in self.noises]
        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        game_state = states.flatten()
        game_next_state = next_states.flatten()
        game_action = actions.flatten()
        self.replay.add((game_state, game_action, rewards, game_next_state, dones))

        if len(self.replay) > self.replay.batch_size:
            self.learn()

    def learn(self):

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

        # Sample a batch of transitions from the replay buffer
        transitions = self.replay.sample()
        game_states, game_actions, rewards, game_next_states, dones = transpose_to_tensor(transitions)

        # Update online critic model
        # Compute actions for next states with the target actor model
        
        game_actions_next = [self.actor_target(torch.split(game_next_states, self.config.state_size, dim=1)[i]) for i in range(self.config.num_agents)]

        game_actions_next = torch.cat(game_actions_next, dim=1)

        # Compute Q values for the next states and next actions with the target critic model
        Q_targets_next = self.critic_target(game_next_states.to(DEVICE), game_actions_next.to(DEVICE))

        # Compute Q values for the current states and actions with the Bellman equation
        game_rewards = rewards.sum(1, keepdim=True)
        game_done = dones.max(1, keepdim=True)[0]
        Q_targets = game_rewards + self.config.discount * Q_targets_next * (1 -game_done)

        # Compute Q values for the current states and actions with the online critic model
        #actions = actions.view(actions.shape[0], -1)
        Q_expected = self.critic_actual(game_states.to(DEVICE), game_actions.to(DEVICE))

        # Compute and minimize the online critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_actual.parameters(), 1)
        self.critic_optimizer.step()

        # Update online actor model
        # Compute actions for the current states with the online actor model
        game_actions_pred = [self.actor_actual(torch.split(game_states, self.config.state_size, dim=1)[i]) for i in range(self.config.num_agents)]
        game_actions_pred = torch.cat(game_actions_pred, dim=1)
        # Compute the online actor loss with the online critic model
        actor_loss = -self.critic_actual(game_states.to(DEVICE), game_actions_pred.to(DEVICE)).mean()
        # Minimize the online critic loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target critic and actor models
        Agent.soft_update(self.actor_target, self.actor_actual, self.config.target_mix)
        Agent.soft_update(self.critic_target, self.critic_actual, self.config.target_mix)

    @staticmethod
    def hard_update(target_model, source_model):
        for target_param, param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def soft_update(target_model, source_model, mix):
        for target_param, online_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - mix) + online_param.data * mix)

    def checkpoint(self,string):
        if not (os.path.isdir(self.config.CHECKPOINT_FOLDER)):
               os.makedirs(self.config.CHECKPOINT_FOLDER)
        torch.save(self.actor_actual.state_dict(), self.config.CHECKPOINT_FOLDER + '/'+string+'_actor.pth')      
        torch.save(self.critic_actual.state_dict(), self.config.CHECKPOINT_FOLDER + '/'+string+'_critic.pth')  
    



