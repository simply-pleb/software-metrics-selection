import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, state_size*8)
        self.fc2 = nn.Linear(state_size*8, state_size*4)
        self.fc3 = nn.Linear(state_size*4, state_size*2)
        self.fc4 = nn.Linear(state_size*2, state_size)
        self.fc5 = nn.Linear(state_size, action_size)
    
    def forward(self, x):
        x_ = F.tanh(self.fc1(x))
        x_ = F.tanh(self.fc2(x_))
        x_ = F.tanh(self.fc3(x_))
        x_ = F.tanh(self.fc4(x_))
        return self.fc5(x_) * (1 - x)  # Q-values for each action