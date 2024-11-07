import torch
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from tqdm import tqdm
import torch.nn as nn

from src.rl.networks import PolicyNetwork, ValueNetwork

class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.01, clip_epsilon=0.2, policy_epochs=10, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.alpha = lr
        self.clip_epsilon = clip_epsilon
        self.policy_epochs = policy_epochs
        self.batch_size = batch_size

        # Actor and Critic networks
        self.actor = PolicyNetwork(state_size, action_size)
        self.critic = ValueNetwork(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=1e-4)

        # Memory to store trajectories
        self.memory = []

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.actor(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
    
    def select_action_deterministic(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.actor(state_tensor)
        dist = Categorical(action_probs)
        action = torch.argmax(action_probs)
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_returns_and_advantages(self, rewards, values, done):
        returns = []
        advantages = []
        G = 0
        A = 0
        for i in reversed(range(len(rewards)-1)):
            G = rewards[i] + (self.gamma * G * (1 - done))
            delta = rewards[i] + (self.gamma * values[i + 1] * (1 - done)) - values[i]
            A = delta + (self.gamma * A * (1 - done))
            returns.insert(0, G)
            advantages.insert(0, A)
        return returns, advantages

    def update(self):
        states, actions, log_probs_old, returns, advantages = zip(*self.memory)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        # Train using multiple epochs and minibatches
        for _ in range(self.policy_epochs):
            for idx in range(0, len(states), self.batch_size):
                batch_states = states[idx:idx + self.batch_size]
                batch_actions = actions[idx:idx + self.batch_size]
                batch_log_probs_old = log_probs_old[idx:idx + self.batch_size]
                batch_returns = returns[idx:idx + self.batch_size]
                batch_advantages = advantages[idx:idx + self.batch_size]

                # Actor (Policy) Loss with Clipping
                new_action_probs = self.actor(batch_states)
                dist = Categorical(new_action_probs)
                batch_log_probs = dist.log_prob(batch_actions)
                ratio = torch.exp(batch_log_probs - batch_log_probs_old)

                # Clipping function
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Critic (Value) Loss
                values = self.critic(batch_states).squeeze()
                # critic_loss = ((values - batch_returns) ** 2).mean()
                critic_loss = nn.functional.smooth_l1_loss(values, batch_returns)
                

                # Backpropagate
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
                self.scheduler_actor.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()
                self.scheduler_critic.step()

        self.memory.clear()

    def train(self, env, num_episodes, max_iters=1000):
        episode_returns = []
        self.scheduler_actor = optim.lr_scheduler.StepLR(
            self.actor_optimizer,
            step_size=200,
            gamma=0.99
        )
        self.scheduler_critic = optim.lr_scheduler.StepLR(
            self.critic_optimizer,
            step_size=200,
            gamma=0.99
        )
        with tqdm(total=num_episodes, desc="Training Episodes") as pbar:
            for episode in range(num_episodes):
                state = env.reset()
                done = False
                episode_return = 0
                self.memory = []

                t = 0
                while not done and t < max_iters:
                    action, log_prob = self.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    episode_return += reward

                    # Store transition in memory
                    value = self.critic(torch.FloatTensor(state)).item()
                    self.store_transition((state, action, log_prob, reward, value, done))

                    state = next_state
                    t += 1

                # Calculate returns and advantages
                rewards, values, dones = zip(*[(trans[3], trans[4], trans[5]) for trans in self.memory])
                returns, advantages = self.compute_returns_and_advantages(rewards, values, done)

                # Update memory with returns and advantages
                self.memory = [(trans[0], trans[1], trans[2], returns[i], advantages[i]) for i, trans in enumerate(self.memory[:-1])]
                
                # PPO Update
                self.update()

                # Track episode return
                actor_lr = self.actor_optimizer.param_groups[0]["lr"]
                critic_lr = self.critic_optimizer.param_groups[0]["lr"]
                pbar.set_postfix(Return=episode_return, ActorLR=actor_lr, CriticLR=critic_lr)
                pbar.update(1)
                episode_returns.append(episode_return)
        return episode_returns


# Example usage
# state_size = 10  # For example, a binary vector of size 10
# action_size = 5  # Number of possible actions
# env = YourEnvironment()  # Replace with your environment
# agent = PPOAgent(state_size, action_size)
# agent.train(env, num_episodes=1000)
