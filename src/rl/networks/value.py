import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, state_size*8)
        self.fc2 = nn.Linear(state_size*8, state_size*4)
        self.fc3 = nn.Linear(state_size*4, state_size*2)
        self.fc4 = nn.Linear(state_size*2, state_size)
        self.fc5 = nn.Linear(state_size, 1) # Output single value

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        value = self.fc5(x)
        return value