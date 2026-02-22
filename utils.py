#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Utility functions including seed configuration
"""
import torch
import numpy as np
import random
import os

from collections import deque

def set_seed(seed):
    """Set random seed for reproducibility"""
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"Random seed set to {seed}")

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.stack(state), np.array(action), np.array(reward, dtype=np.float32),
                np.stack(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

