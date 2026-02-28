#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
Create and set configurations of the environment
Optional intrinsic motivation
"""
import ale_py
import gymnasium as gym
import numpy as np

from gymnasium.wrappers import TransformReward, RecordEpisodeStatistics
from wrappers import MetricsWrapper, RawRewardTracker, HullWrapper

gym.register_envs(ale_py)

def make_env_with_metrics(name, 
                          render_mode=None, 
                          use_ram=True, 
                          agent_style='Vanilla', 
                          clip_rewards=False):
    
    env = gym.make(f"ALE/{name}-v5", 
                   render_mode=render_mode,
                   obs_type="ram" if use_ram else "rgb",
                   frameskip=4)

    # Track raw rewards before any modification
    env = RawRewardTracker(env)    
    env = RecordEpisodeStatistics(env)
    
    X_BYTE = 10  
    Y_BYTE = 16  
    PELLET_BYTE = 119      
    env = MetricsWrapper(env, X_BYTE, Y_BYTE, PELLET_BYTE)

    if agent_style == 'Hull':
        env = HullWrapper(env)
    
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


# In[4]:


"""
# Quick test
env = make_env_with_metrics("MsPacman", use_ram=True)
state, _ = env.reset()

total_reward = 0
for i in range(1000):
    action = env.action_space.sample()
    state, reward, term, trunc, info = env.step(action)
    total_reward += reward
    
    if term or trunc:
        if 'metrics' in info:
            print("\nMetrics:")
            print(f"  Pellets eaten: {info['metrics']['pellets_eaten']}")
            print(f"  Power pellets: {info['metrics']['power_pellets_eaten']}")
            print(f"  Ghosts eaten: {info['metrics']['ghosts_eaten']}")
            print(f"  External reward: {info['metrics']['external_reward']}")
        break

print(f"Shaped reward (what agent saw): {total_reward}")
"""


# In[ ]:




