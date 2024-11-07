import torch
import torch.nn as nn
import torch.nn.functional as F

class StateActionValueNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256):
        super(StateActionValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)