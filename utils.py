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

def compute_directional_pellet_salience(pacman_x, pacman_y, traversable_positions, eaten_pellet_positions, n_steps=10, n_actions=5):
    # vai considerar as paredes como salientes, quão ruim é isto? 
    """
    For each action direction, estimate how many uneaten pellets
    are in the next n_steps positions.
    
    eaten_pellet_positions: set of (x,y) where pellets were eaten
    Returns C of shape [n_actions]
    """
    # [noop, up, right, left, down]
    directions = [(0, 0), (0, -1), (1, 0), (-1, 0), (0, 1)]
    
    # Ms. Pac-Man maze approximate bounds
    X_MIN, X_MAX = 0, 160 # CONFIRAR NA RAM ESSES VALORES 
    Y_MIN, Y_MAX = 0, 210
    STEP_SIZE = 2  # approximate pixels per step (alternate 2 and 3)
    
    C = np.zeros(n_actions)
    
    for i, (dx, dy) in enumerate(directions):
        if dx == 0 and dy == 0:  # noop
            C[i] = 0.0
            continue
        
        pellet_score = 0.0
        for step in range(1, n_steps + 1):
            # circular wrapping for maze tunnels (left-right wrap)
            next_x = (pacman_x + dx * STEP_SIZE * step - X_MIN) % (X_MAX - X_MIN) + X_MIN # só estou na dúvida do stepsize 
            next_y = np.clip(pacman_y + dy * STEP_SIZE * step, Y_MIN, Y_MAX)  # no vertical wrap - clip pq não passa dos limites
            
            pos = (int(next_x), int(next_y))
            
            # discount further positions
            discount = 1.0 / step
            
            if pos in traversable_positions and pos not in eaten_pellet_positions:
                pellet_score += discount  # uneaten → attractive
        
        C[i] = pellet_score
    
    return C

