import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque

from tqdm import tqdm

from src.rl.networks import PolicyNetwork


# Define the REINFORCE agent
class REINFORCEAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, lr_decay=0.999):
        self.policy = PolicyNetwork(state_size, action_size)
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        # self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: lr_decay ** epoch)
        # self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=lr/100, end_factor=lr)
        self.lr = lr
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def store_outcome(self, log_prob, reward):
        # Store log probabilities and rewards for an episode
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy(self):
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)

        # Normalize rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-9
        )

        # Compute policy loss
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Step the learning rate scheduler
        self.scheduler.step()

        # Reset storage for the next episode
        self.log_probs = []
        self.rewards = []
    
    def update_policy_reduced_variance(self, entropy_coeff=1e-4):
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)

        # # Normalize rewards
        # discounted_rewards = torch.tensor(discounted_rewards)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
        #     discounted_rewards.std() + 1e-9
        # )

        # Compute policy loss with entropy regularization
        policy_loss = []
        entropy_loss = 0
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
            entropy_loss += -log_prob.exp() * log_prob  # Entropy

        policy_loss = torch.cat(policy_loss).sum() - entropy_coeff * entropy_loss.sum()

        # # L2 regularization
        # l2_reg = torch.tensor(0.)
        # for param in self.policy.parameters():
        #     l2_reg += torch.norm(param) ** 2
        # policy_loss += l2_lambda * l2_reg

        # Update policy with gradient clipping
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Step the learning rate scheduler
        self.scheduler.step()

        # Reset storage for the next episode
        self.log_probs = []
        self.rewards = []


    def select_action(self, state):
        # Select an action based on policy probabilities
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy.forward(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def select_action_deterministic(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.policy(state_tensor)
        dist = torch.distributions.Categorical(action_probs)
        action = torch.argmax(action_probs)
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    # Main loop for REINFORCE
    def train(self, env, num_episodes, max_steps=50):
        episode_returns = []
        # self.scheduler.total_iters = num_episodes
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=50,
            gamma=0.99
        )
        with tqdm(total=num_episodes, desc="Training Episodes") as pbar:
            for episode in range(num_episodes):
                state = env.reset()
                episode_return = 0
                done = False
                t = 0
                while not done and t < max_steps:
                    action, log_prob = self.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    self.store_outcome(log_prob, reward)
                    state = next_state
                    episode_return += reward
                    t += 1

                # Update the policy at the end of each episode
                # self.update_policy()
                self.update_policy_reduced_variance()
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(Return=episode_return, LR=current_lr)
                pbar.update(1)
                episode_returns.append(episode_return)

        return episode_returns


# Hyperparameters
# input_dim = 4  # Example for CartPole-v0
# hidden_dim = 128
# output_dim = 2
# lr = 1e-3
# gamma = 0.99
# num_episodes = 1000

# Environment setup (e.g., OpenAI Gym's CartPole)
# Uncomment the following line if using an environment like CartPole
# import gym
# env = gym.make('CartPole-v0')

# Initialize agent
# agent = REINFORCEAgent(input_dim, hidden_dim, output_dim, lr, gamma)

# Training loop
# Uncomment the following line if using an environment
# reinforce_train(env, agent, num_episodes)
