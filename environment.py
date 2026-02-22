#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Create and set configurations of the environment
"""
import ale_py
import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TransformReward, RecordEpisodeStatistics
from wrappers import MetricsWrapper

gym.register_envs(ale_py)

def make_env_with_metrics(name, render_mode=None, use_ram=True, clip_rewards=False):
    """Environment with metrics tracking and optional reward shaping"""
    
    env = gym.make(f"ALE/{name}-v5", 
                   render_mode=render_mode,
                   obs_type="ram" if use_ram else "rgb",
                   frameskip=4)
    
    env = RecordEpisodeStatistics(env)
    
    X_BYTE = 10  
    Y_BYTE = 16  
    PELLET_BYTE = 119  
    
    env = MetricsWrapper(env, X_BYTE, Y_BYTE, PELLET_BYTE)
    
    if clip_rewards:
        env = TransformReward(env, lambda r: np.sign(r))  # Clip
    else:
        env = apply_reward_shaping(env)  # Shape
    
    return env

def scale_reward(r):
    """Transform rewards to reasonable range"""
    if r <= 0:
        return r # negative reward
    elif r == 10:
        return 1 # normal pellet
    elif r == 50:
        return 3 # power pellet
    elif r >= 200:
        return r / 50 # ghosts
    else:
        return r / 100 # fruits and other bonuses


def apply_reward_shaping(env):
    """Apply reward shaping to environment"""
    return TransformReward(env, scale_reward)

