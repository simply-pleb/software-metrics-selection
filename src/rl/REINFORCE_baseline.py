import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from tqdm import tqdm

from src.rl.networks import PolicyNetwork, ValueNetwork


class REINFORCEWithBaselineAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, lr_decay=0.999):
        self.reward_mean = 0.0
        self.reward_variance = 1.0
        self.running_count = 0  # To track the number of rewards seen
        self.policy = PolicyNetwork(state_size, action_size)
        self.value = ValueNetwork(state_size)  # Value network for baseline
        self.policy_optimizer = optim.RMSprop(
            self.policy.parameters(), lr=lr, weight_decay=1e-4
        )
        self.value_optimizer = optim.RMSprop(
            self.value.parameters(), lr=lr / 10, weight_decay=1e-4
        )
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []
        self.states = []

    def update_reward_statistics(self, rewards):
        for reward in rewards:
            self.running_count += 1
            delta = reward - self.reward_mean
            self.reward_mean += delta / self.running_count
            delta2 = reward - self.reward_mean
            self.reward_variance += delta * delta2

    def store_outcome(self, state, log_prob, reward):
        # Store states, log probabilities, and rewards for an episode
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

    def update_policy_and_value(self):
        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Convert states to tensor and compute baseline values
        states_tensor = torch.tensor(self.states, dtype=torch.float32)
        baseline_values = self.value(states_tensor).squeeze(-1)

        # print(f"Baseline Values Shape: {baseline_values.shape}; {baseline_values}")
        # print(f"Discounted Rewards Shape: {discounted_rewards.shape}; {discounted_rewards}")

        # Compute advantages
        advantages = discounted_rewards - baseline_values.detach()

        # Compute policy loss
        policy_loss = []
        for log_prob, advantage in zip(self.log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.cat(policy_loss).sum()

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Compute value loss (MSE between discounted rewards and baseline values)
        value_loss = F.mse_loss(baseline_values, discounted_rewards)

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Reset storage for the next episode
        self.log_probs = []
        self.rewards = []
        self.states = []

    # def update_policy_and_value(self, lambda_=0.95):
    #     # Convert states and rewards to tensors
    #     states_tensor = torch.tensor(self.states, dtype=torch.float32)
    #     rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)

    #     # Compute baseline values
    #     baseline_values = self.value(states_tensor).squeeze(-1)

    #     # Calculate GAE advantages
    #     advantages = []
    #     gae = 0
    #     for t in reversed(range(len(rewards_tensor))):
    #         delta = rewards_tensor[t] + self.gamma * (baseline_values[t + 1] if t + 1 < len(rewards_tensor) else 0) - baseline_values[t]
    #         gae = delta + self.gamma * lambda_ * gae
    #         advantages.insert(0, gae)
    #     # advantages = torch.tensor(advantages, dtype=torch.float32)
    #     advantages = torch.tensor(advantages, dtype=torch.float32)
    #     # print(self.reward_mean, (np.sqrt(self.reward_variance / self.running_count) + 1e-8))
    #     # advantages = (advantages - self.reward_mean) / (np.sqrt(self.reward_variance / self.running_count) + 1e-8)

    #     # Compute policy loss
    #     policy_loss = []
    #     for log_prob, advantage in zip(self.log_probs, advantages):
    #         policy_loss.append(-log_prob * advantage)
    #     policy_loss = torch.cat(policy_loss).sum()

    #     # Update policy network
    #     self.policy_optimizer.zero_grad()
    #     policy_loss.backward()
    #     self.policy_optimizer.step()

    #     # Compute value loss (MSE between target values and baseline values)
    #     # Target values are the GAE-adjusted rewards
    #     target_values = advantages + baseline_values.detach()
    #     value_loss = F.mse_loss(baseline_values, target_values)

    #     # Update value network
    #     self.value_optimizer.zero_grad()
    #     value_loss.backward()
    #     self.value_optimizer.step()

    #     # Reset storage for the next episode
    #     self.log_probs = []
    #     self.rewards = []
    #     self.states = []

    def select_action(self, state):
        # Select an action based on policy probabilities
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state_tensor)
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

    def train(self, env, num_episodes, max_steps=50):
        episode_returns = []
        # self.reward_mean = 0.0
        # self.reward_variance = 1.0
        # self.running_count = 0  # To track the number of rewards seen
        with tqdm(total=num_episodes, desc="Training Episodes") as pbar:
            for episode in range(num_episodes):
                state = env.reset()
                episode_return = 0
                done = False
                t = 0
                while not done and t < max_steps:
                    action, log_prob = self.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    self.store_outcome(state, log_prob, reward)
                    state = next_state
                    episode_return += reward
                    t += 1

                # Update the policy and value network at the end of each episode
                # self.update_reward_statistics([reward])
                self.update_policy_and_value()
                pbar.set_postfix(Return=episode_return)
                pbar.update(1)
                episode_returns.append(episode_return)

        return episode_returns
