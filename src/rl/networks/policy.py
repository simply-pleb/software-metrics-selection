import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, state_size)
        # self.fc2 = nn.Linear(state_size, state_size)
        # self.fc3 = nn.Linear(state_size*4, state_size*2)
        # self.fc4 = nn.Linear(state_size*2, state_size)
        self.fc5 = nn.Linear(state_size, action_size)
        
        # Initialize weights to achieve zero mean and low standard deviation
        # for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
        # for layer in [self.fc1, self.fc2, self.fc5]:
        for layer in [self.fc1, self.fc5]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # Low standard deviation
            nn.init.constant_(layer.bias, 0.0)  # Zero mean for initial activations

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        # x = F.relu(self.fc1(state))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc5(x)
        action_probs = F.softmax(x, dim=-1)  # Output probabilities
        return action_probs * (1 + 1e-7 - state)