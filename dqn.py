#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
DQN network architecture for RAM-based observations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN_RAM(nn.Module):
    """DQN for RAM-based observations (128 bytes input)"""
    
    def __init__(self, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

