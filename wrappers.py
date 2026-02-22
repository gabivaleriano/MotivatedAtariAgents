#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Custom Gymnasium wrappers for Ms. Pac-Man
"""
import gymnasium as gym
import numpy as np


class RawRewardTracker(gym.Wrapper):
    """Track raw rewards before any transformation"""
    
    def __init__(self, env):
        super().__init__(env)
        self.episode_raw_rewards = []
        self.cumulative_raw_reward = 0
    
    def reset(self, **kwargs):
        """Reset episode tracking"""
        self.episode_raw_rewards = []
        self.cumulative_raw_reward = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step environment and track raw reward"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store raw reward BEFORE any shaping
        self.episode_raw_rewards.append(reward)
        self.cumulative_raw_reward += reward
        
        # Add to info at episode end
        if terminated or truncated:
            info['raw_episode_return'] = self.cumulative_raw_reward
            info['raw_rewards_list'] = self.episode_raw_rewards.copy()
        
        return obs, reward, terminated, truncated, info


class MetricsWrapper(gym.Wrapper):
    """Wrapper that tracks RAM state and calculates metrics"""
    
    def __init__(self, env, x_byte, y_byte, pellet_byte):
        super().__init__(env)
        self.x_byte = x_byte
        self.y_byte = y_byte
        self.pellet_byte = pellet_byte
        
        # Episode tracking
        self.episode_positions = []
        self.episode_steps = 0
        self.episode_pellets_start = 0
        self.current_level = 1
        
    def reset(self, **kwargs):
        """Reset episode tracking"""
        obs = self.env.reset(**kwargs)
        
        self.episode_positions = []
        self.episode_steps = 0
        
        # Get RAM state
        if isinstance(obs, tuple):
            ram_state = obs[0]
        else:
            ram_state = obs
        
        self.episode_pellets_start = int(ram_state[self.pellet_byte])
        
        return obs
    
    def step(self, action):
        """Step environment and track metrics"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store position (from RAM)
        x = int(obs[self.x_byte])
        y = int(obs[self.y_byte])
        self.episode_positions.append((x, y))
        
        self.episode_steps += 1
        
        # Detect level completion
        current_pellets = int(obs[self.pellet_byte])
        
        # If pellet count increased (new level started)
        if current_pellets > self.episode_pellets_start:
            self.current_level += 1
            self.episode_pellets_start = current_pellets  # FIX: Update baseline
        
        # Get raw rewards from RawRewardTracker (in info)
        raw_rewards = info.get('raw_rewards_list', [])
        
        # Calculate metrics at episode end
        if terminated or truncated:
            metrics = self.calculate_metrics(raw_rewards)
            info['metrics'] = metrics
        
        return obs, reward, terminated, truncated, info
    
    def calculate_metrics(self, raw_rewards):
        """Calculate all metrics for the episode"""
        
        # 1. Average Lifetime
        lifetime = self.episode_steps
        
        # 2. Pellet Efficiency (from RAW rewards)
        pellets_eaten = sum(1 for r in raw_rewards if r == 10)
        pellet_efficiency = pellets_eaten / lifetime if lifetime > 0 else 0
        
        # 3. Ghost-Eating Efficiency (from RAW rewards)
        power_pellets_eaten = sum(1 for r in raw_rewards if r == 50)
        
        ghosts_eaten = 0
        frightened_mode = False
        frightened_timer = 0
        
        for r in raw_rewards:
            # Enter frightened mode when power pellet eaten
            if r == 50:
                frightened_mode = True
                frightened_timer = 0
            
            # Count ghosts (unambiguous rewards)
            if r in [400, 800, 1600]:
                ghosts_eaten += 1
            # For 200 points, only count during frightened mode
            elif r == 200 and frightened_mode:
                ghosts_eaten += 1
            
            # Frightened mode lasts ~40 steps
            if frightened_mode:
                frightened_timer += 1
                if frightened_timer > 40:
                    frightened_mode = False
        
        ghost_efficiency = ghosts_eaten / power_pellets_eaten if power_pellets_eaten > 0 else 0
        
        # 4. Backtracking Rate
        backtrack_count = 0
        visited_positions = set()
        
        for pos in self.episode_positions:
            if pos in visited_positions:
                backtrack_count += 1
            visited_positions.add(pos)
        
        backtrack_rate = backtrack_count / lifetime if lifetime > 0 else 0
        
        # 5. External (raw) reward
        external_reward = sum(raw_rewards) if raw_rewards else 0
        
        return {
            'lifetime': lifetime,
            'pellet_efficiency': pellet_efficiency,
            'ghost_eating_efficiency': ghost_efficiency,
            'backtracking_rate': backtrack_rate,
            'max_level_reached': self.current_level,
            'external_reward': external_reward,
            # Additional info
            'pellets_eaten': pellets_eaten,
            'power_pellets_eaten': power_pellets_eaten,
            'ghosts_eaten': ghosts_eaten,
        }

