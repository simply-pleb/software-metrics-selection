import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, state_size)
        self.fc2 = nn.Linear(state_size, state_size)
        # self.fc3 = nn.Linear(state_size//4, state_size//8)
        # self.fc4 = nn.Linear(state_size/8, state_size)
        self.fc5 = nn.Linear(state_size, 1) # Output single value
        self.dropout = nn.Dropout(p=0.5)  # 50% dropout
        for layer in [self.fc1, self.fc2, self.fc5]:
        # for layer in [self.fc1, self.fc5]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)  # Low standard deviation
            nn.init.constant_(layer.bias, 0.0)  # Zero mean for initial activations

    def forward(self, state):
        x = F.tanh(self.fc1(state))
        # x = self.dropout(x)  # Apply dropout during training
        x = F.tanh(self.fc2(x))
        # x = self.dropout(x)  # Apply dropout during training
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        value = self.fc5(x)
        return value