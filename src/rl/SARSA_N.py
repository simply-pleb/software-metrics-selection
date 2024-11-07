import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import SmoothL1Loss
# from torch.nn import _reduction as _Reduction, functional as F
import numpy as np
import random
from collections import deque

from tqdm import tqdm

from src.rl.networks import QNetwork


class SemiGradientNSarsa:
    def __init__(self, state_size, action_size, n_steps=5, alpha=0.01, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.n_steps = n_steps
        self.alpha = alpha
        self.gamma = gamma
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.memory = []
        self.loss_function = SmoothL1Loss()

    def select_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_size)  # Explore
        else:
            with torch.no_grad():
                q_values = self.q_network(torch.FloatTensor(state))
                return torch.argmax(q_values).item()  # Exploit
    
    def reset_memory(self):
        self.memory = []
    
    def store_transition(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.n_steps:
            self.memory.pop(0)

    def update_q_values(self, done):
        if len(self.memory) == 0:
            return

        # Calculate returns
        states = []
        actions = []
        rewards = []
        next_states = []
        next_actions = []
        for i in range(len(self.memory)):
            state, action, reward, next_state, next_action = self.memory[i]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            next_actions.append(next_action)

        G = 0
        n = self.n_steps# min(len(rewards), self.n_steps)
        for i in range(n):
            G += (self.gamma ** i) * rewards[i]
        if not done:
            next_state = next_states[-1]  # Use the last state to estimate Q
            next_action = next_actions[-1]  # Use the last state to estimate Q
            with torch.no_grad():
                next_q_values = self.q_network(torch.FloatTensor(next_state))
                G += (self.gamma ** n) * next_q_values[next_action].item()

        # Update the Q-values
        state_tensor = torch.FloatTensor(states[0])
        action_tensor = torch.LongTensor([actions[0]])
        target = torch.FloatTensor([G])
        predicted = self.q_network(state_tensor)[action_tensor]

        # loss = F.smooth_l1_loss(predicted, predicted)#(predicted - target) ** 2
        loss = self.loss_function(predicted, target)
        print(loss)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env, num_episodes, max_iter=1000, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        epsilon = epsilon_start
        episode_returns = []
        with tqdm(total=num_episodes, desc="Training Episodes") as pbar:
            for episode in range(num_episodes):
                state = env.reset()
                done = False
                episode_return = 0
                self.reset_memory()
                t = 0
                while not done and t < max_iter:
                    action = self.select_action(state, epsilon)
                    next_state, reward, done, _ = env.step(action)
                    # store SARSA
                    self.store_transition((state, action, reward, next_state, self.select_action(next_state, epsilon)))
                    episode_return += reward
                    
                    if t - self.n_steps + 1 > 0:
                        self.update_q_values(done)
                    if done:
                        break
                    state = next_state
                    t += 1

                # Decay epsilon
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                # tqdm.write(f"Episode {episode + 1}: Return = {episode_return}")
                pbar.set_postfix(Return=episode_return)
                pbar.update(1)
                episode_returns.append(episode_return)
        return episode_returns


# Example usage (assuming you have an environment):
# state_size = 10  # For example, a binary vector of size 10
# action_size = 5  # Number of possible actions
# env = YourEnvironment()  # Replace with your environment
# agent = SemiGradientNSarsa(state_size, action_size)
# agent.train(env, num_episodes=1000)