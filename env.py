#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from wrappers import RestrictActionsWrapper, MetricsWrapper, LifeLossWrapper
import ale_py
import gymnasium as gym
from gymnasium.wrappers import TransformReward


# Register Atari (ALE) environments with gymnasium
gym.register_envs(ale_py)

def make_env_with_metrics(seed):
        
    env = gym.make("ALE/MsPacman-v5", obs_type="rgb", frameskip=4)  # Create the Ms. Pac-Man game environment     
    env = apply_reward_shaping(env)
    env = LifeLossWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env) # Wrapper that records episode statistics    
    env = MetricsWrapper(env)   
    env = gym.wrappers.ResizeObservation(env, (84, 84)) # Resize each frame to 84x84 pixels 
    env = gym.wrappers.GrayscaleObservation(env) # Convert frames from RGB to grayscale
    env = gym.wrappers.FrameStackObservation(env, 4) # Stack the last 4 frames as input channels  
    env = MaxAndSkipEnv(env, skip=4)    # Repeat each action for 4 frames and take the pixel-wise max between the last 2 frames of the skip
    env = RestrictActionsWrapper(env)
    
    env.action_space.seed(seed)
    return env

# Apply a reward transformation
def scale_reward(r):
    """Transform rewards to reasonable range"""
    if r <= 0:
        return r # negative reward
    else:# r >= 10:
        return r / 10 

def apply_reward_shaping(env):
    """Apply reward shaping to environment"""
    return TransformReward(env, scale_reward)


# In[ ]:




