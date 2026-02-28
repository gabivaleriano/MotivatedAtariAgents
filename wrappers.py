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


# In[ ]:


class HullWrapper(gym.Wrapper):

    def __init__(self, env, 
                 lambda_wanting=1.0,    # Drive reduction weight (Hull component)
                 hunger_inc=0.2, 
                 max_hunger=10.0):
        super().__init__(env)
        self.lambda_wanting = lambda_wanting
        self.hunger_inc = hunger_inc
        self.max_hunger = max_hunger
        self.hunger = 0.0
        self.last_score = 0

        self.intrinsic_total = 0
        
        # Step-level tracking (history within episode)
        self.step_history = {
            'hunger': [],
            'wanting': [],
            'intrinsic': [],
            'extrinsic': []
        }
        
        # Cross-episode tracking (for analyzing trends across episodes)
        self.episode_history = {
            'want_total': [],
            'intrinsic_total': [],
            'extrinsic_total': [],
            'episode_length': [],
            'pellets_eaten': []
        }
        
        self.current_step = 0
        self.current_episode = 0
        self.pellets_this_episode = 0
    
    def reset(self, **kwargs):
        # Save episode history before reset
        if self.current_step > 0:  # Not first reset
            self.episode_history['want_total'].append(self.want_total)
            self.episode_history['intrinsic_total'].append(self.intrinsic_total)
            self.episode_history['extrinsic_total'].append(self.episode_extrinsic_reward)
            self.episode_history['episode_length'].append(self.current_step)
            self.episode_history['pellets_eaten'].append(self.pellets_this_episode)
        
        # Reset episode-level trackers
        self.hunger = 0.0
        self.episode_extrinsic_reward = 0.0
        self.want_total = 0
        self.intrinsic_total = 0
        self.current_step = 0
        self.pellets_this_episode = 0
        
        # Reset step history
        self.step_history = {
            'hunger': [],
            'wanting': [],
            'intrinsic': [],
            'extrinsic': [],
            'pellet_eaten': []
        }
        
        obs, info = self.env.reset(**kwargs)
        self.last_score = info.get("score", 0) or 0
        self.current_episode += 1
        
        return obs, info

   
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Detect pellet consumption
        score_now = info.get("score", 0) or 0
        #pellet_eaten = score_now > self.last_score
        pellet_eaten = (reward == 10)
        self.last_score = score_now
        
        if pellet_eaten:
            self.pellets_this_episode += 1
        
        old_hunger = self.hunger
    
        self.hunger += self.hunger_inc
        self.hunger = min(self.hunger, self.max_hunger)
        
        # Initialize reward components
        wanting_reward = 0.0

        
        if pellet_eaten:
            self.hunger *= 0.2  # Satiation             
            drive_reduction = old_hunger - self.hunger
            
        wanting_reward = -self.lambda_wanting * self.hunger                
        
        # Total intrinsic reward combines both systems
        intrinsic_reward = wanting_reward 
        
        # Total reward (for learning)
        total_reward = reward + intrinsic_reward
        
        # Accumulate episode totals
        self.episode_extrinsic_reward += reward
        self.want_total += wanting_reward
        self.intrinsic_total += intrinsic_reward
        
        # Record step-level data
        self.step_history['hunger'].append(self.hunger)
        self.step_history['wanting'].append(wanting_reward)
        self.step_history['intrinsic'].append(intrinsic_reward)
        self.step_history['extrinsic'].append(reward)
        self.step_history['pellet_eaten'].append(pellet_eaten)
        
        self.current_step += 1
        
        # On episode end, store extrinsic total in info
        if term or trunc:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["r_extrinsic"] = self.episode_extrinsic_reward
            info["episode"]["want"] = self.want_total
            info["episode"]["intrinsic"] = self.intrinsic_total
            info["episode"]["step_history"] = self.step_history.copy()  # Include step history
            info["episode"]["pellets_eaten"] = self.pellets_this_episode
            
        # Track all components for analysis
        info["hunger_drive"] = self.hunger
        info["wanting_reward"] = wanting_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["current_step"] = self.current_step
        info["current_episode"] = self.current_episode
       
        return obs, total_reward, term, trunc, info
    
    def get_episode_history(self):
        """Get history across all episodes"""
        return self.episode_history.copy()
    
    def get_current_step_history(self):
        """Get step-by-step history for current episode"""
        return self.step_history.copy()

print("Hull wrapper ready ✓")


# In[ ]:


class HullIntrinsicMotivation(gym.Wrapper):
    """
    Hull-style intrinsic motivation
    """
    def __init__(self, env, curiosity_weight=0.1):
        super().__init__(env)

    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate intrinsic reward (curiosity)
        intrinsic_reward = self._calculate_curiosity(obs)
        
        # Add intrinsic reward to extrinsic reward
        total_reward = reward + self.curiosity_weight * intrinsic_reward
        
        # Store intrinsic reward in info for tracking
        info['intrinsic_reward'] = intrinsic_reward
        info['extrinsic_reward'] = reward
        
        return obs, total_reward, terminated, truncated, info

