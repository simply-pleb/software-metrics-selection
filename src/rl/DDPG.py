import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tqdm import tqdm
import random
import copy

from src.rl.networks import PolicyNetwork, StateActionValueNetwork


class ReplayBuffer:
    def __init__(self, buffer_size=100000, batch_size=64):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        samples = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.float),
            torch.tensor(rewards, dtype=torch.float).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, state_size, action_size, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005, lr_decay=0.99):
        # Actor and critic networks
        self.actor = PolicyNetwork(state_size, action_size)
        self.critic = StateActionValueNetwork(state_size, action_size)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=lambda epoch: lr_decay ** epoch)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=lambda epoch: lr_decay ** epoch)

        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer()
        
        # Noise process
        self.noise_std = 0.2
        self.noise_decay = 0.995

    def select_action(self, state, noise=True):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        action = self.actor(state).detach().squeeze().numpy()
        if noise:
            action += self.noise_std * np.random.normal(size=action.shape)
        return np.clip(action, -1, 1)

    def store_outcome(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_networks(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Critic loss
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            target_q_values = rewards + self.gamma * (1 - dones) * next_q_values
        
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, target_q_values)
        
        # Update critic network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Step the learning rate schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def train(self, env, num_episodes, batch_size=64):
        episode_returns = []
        with tqdm(total=num_episodes, desc="Training Episodes") as pbar:
            for episode in range(num_episodes):
                state = env.reset()
                episode_return = 0
                done = False

                while not done:
                    action = self.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    self.store_outcome(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    # Perform updates
                    self.update_networks(batch_size=batch_size)
                
                self.noise_std *= self.noise_decay  # Decay exploration noise
                pbar.set_postfix(Return=episode_return, LR_Actor=self.actor_optimizer.param_groups[0]['lr'], LR_Critic=self.critic_optimizer.param_groups[0]['lr'])
                pbar.update(1)
                episode_returns.append(episode_return)
                
        return episode_returns
