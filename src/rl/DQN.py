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

# Define the DQN Agent
class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        device,
        gamma=0.99,
        lr=0.00001,
        batch_size=64,
        buffer_size=10000,
        target_update=10,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device

        # Initialize main and target networks
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Set up optimizer and replay memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.steps_done = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice(range(self.action_size))
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            return torch.argmax(self.policy_net(state)).item()

    def store_transition(self, transition):
        self.memory.append(transition)

    def sample_batch(self):
        return random.sample(self.memory, self.batch_size)

    def update_q_values(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.sample_batch()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Calculate Q(s, a) using policy network
        current_q_values = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        )

        # Calculate target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Update target network every target_update steps
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(
        self,
        env,
        num_episodes,
        max_iters=None,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.995,
    ):
        epsilon = epsilon_start
        returns = []
        # max_iters = self.state_size
        max_iters = max_iters if max_iters is not None else self.state_size
        print(max_iters)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=1000,
            gamma=0.99
        )
        with tqdm(total=num_episodes, desc="Training DQN") as pbar:
            for episode in range(num_episodes):           
                state = env.reset()
                done = False
                episode_return = 0
                t = 0
                while not done and t < max_iters:
                    action = self.select_action(state, epsilon)
                    next_state, reward, done, _ = env.step(action)

                    # Store transition in memory
                    self.store_transition((state, action, reward/max_iters, next_state, done))

                    # Update Q values
                    self.update_q_values()

                    # Move to the next state
                    state = next_state
                    episode_return += reward
                    self.steps_done += 1
                    t += 1

                # Decay epsilon
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
                returns.append(episode_return)
                # tqdm.write(f"Episode {episode + 1}: Return = {episode_return:.2f}")
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix(Return=episode_return, LR=current_lr)
                pbar.update(1)

        return returns