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
        self.last_raw_reward = 0

        # Cross-episode tracking (for analyzing trends across episodes)
        self.episode_history = {
            'extrinsic_total': [],
        }
    
    def reset(self, **kwargs):
        """Reset episode tracking"""

        # Save episode history before reset
        if len(self.episode_raw_rewards) > 0:
            self.episode_history['extrinsic_total'].append(self.episode_raw_rewards)
            self.episode_raw_rewards = []
            self.cumulative_raw_reward = 0
            
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step environment and track raw reward"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store raw reward BEFORE any shaping
        self.episode_raw_rewards.append(reward)
        self.last_raw_reward = reward
        self.cumulative_raw_reward += reward
        
        
        # Add to info at episode end
        if terminated or truncated:
            info['raw_episode_return'] = self.cumulative_raw_reward
            info['raw_rewards_list'] = self.episode_raw_rewards.copy()

        return obs, reward, terminated, truncated, info

class MetricsWrapper(gym.Wrapper):
    """Wrapper that tracks RAM state and calculates metrics"""
    
    def __init__(self, env, raw_tracker=None):
        super().__init__(env)
        x_byte = 10
        y_byte = 16
        pellet_byte = 119

        self.raw_tracker = raw_tracker
        
        # Episode tracking
        self.episode_positions = []
        self.episode_steps = 0
        self.pellets_eaten = 0
        self.past_119 = 0                       # past value of byte 119 - pellets
        self.x_byte = 10
        self.y_byte = 16
        self.pellet_byte = 119

        # Level tracking
        self.current_episode_level = 0          # resets each episode, max 8

        
    def reset(self, **kwargs):
        """Reset episode tracking"""
        obs = self.env.reset(**kwargs)        
        self.episode_positions = []
        self.episode_steps = 0
        self.pellets_eaten = 0      # add this
        self.past_119 = 0    
        self.current_episode_level = 0    
        
        # Get RAM state
        if isinstance(obs, tuple): # if obs is a tuple like (array, dict
            ram_state = obs[0] # grab just the first element (the array)
        else:
            ram_state = obs
        
        return obs
    
    def step(self, action):
        """Step environment and track metrics"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store position (from RAM)
        x = int(obs[self.x_byte])
        y = int(obs[self.y_byte])
        self.episode_positions.append((x, y))
        
        self.episode_steps += 1

        current_119 = int(obs[self.pellet_byte])
       
        # If self.pellet_byte == 0 indicates level start. 
        if current_119 == 0 and self.past_119 != 0:
            self.current_episode_level += 1

        # If 119 changed == pellet eaten
        if current_119 - self.past_119 == 1:
            self.pellets_eaten += 1
            self.past_119 = current_119
            
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
        pellets_eaten = self.pellets_eaten
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
            'level_reached': self.current_episode_level,
            'external_reward': external_reward,
            'pellets_eaten': pellets_eaten,
            'power_pellets_eaten': power_pellets_eaten,
            'ghosts_eaten': ghosts_eaten,
        }


# In[ ]:


class HullWrapper(gym.Wrapper):
    
    def __init__(self, env, raw_tracker=None):
        super().__init__(env)
        self.raw_tracker = raw_tracker
        self.D = 30          # start at homeostasis
        self.D_star = 30     # homeostasis level
        self.D_max = 50
        self.D_min = 0
        self.prev_pos = (85, 98)

        # Step-level tracking (history within episode)
        self.current_episode = 0
        self.step_history = { 'drive': []}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        curr_pos = (int(obs[10]), int(obs[16]))   # x_byte=10, y_byte=16

        # 1. detect eating first (takes priority)
        raw_reward = self.raw_tracker.last_raw_reward if self.raw_tracker else 0
        if raw_reward in [10, 50, 200, 400, 800, 1600]:
            energy_delta = +3
        elif curr_pos != self.prev_pos:
            energy_delta = -2
        else:
            energy_delta = -1

        self.prev_pos = curr_pos            

        # 2. update drive
        self.D = np.clip(self.D + energy_delta, self.D_min, self.D_max)

        # 3. compute intrinsic reward
        if self.D < self.D_star:
            Ri = -((self.D_star - self.D) / self.D_star) ** 2
        else:
            Ri = (self.D - self.D_star) / self.D_star  # note: no penalty per spec

        self.step_history['drive'].append(self.D)

        if terminated or truncated:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["D"] = self.D
            info["episode"]["step_history"] = self.step_history.copy()
        
        # Keep ONLY this one - CombineRewardWrapper needs it
        info["drive_reward"] = Ri
        
        return obs, reward, terminated, truncated, info


    def reset(self, **kwargs):
       
        # Reset episode-level trackers
        self.D = 30          # start at homeostasis
        self.prev_pos = (85, 98)

        # Step-level tracking (history within episode)
        self.step_history = { 'drive': []}
        
        obs, info = self.env.reset(**kwargs)
        self.current_episode += 1
        
        return obs, info  

class CombineRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        intrinsic = info.get("drive_reward", 0.0)
        total = reward + intrinsic
        
        info["extrinsic_reward"] = reward
        info["intrinsic_reward"] = intrinsic
        info["total_reward"] = total
        
        return obs, total, term, trunc, info
  


# In[ ]:


class WantLikeWrapper(gym.Wrapper):

    def __init__(self, env, 
                 lambda_wanting=1.0,    # Drive reduction weight (Hull component)
                 hunger_inc=0.2, 
                 max_hunger=10.0):
        
        super().__init__(env)
        self.lambda_wanting = lambda_wanting
        self.hunger_inc = hunger_inc
        self.max_hunger = max_hunger
        self.hunger = 0.0
        self.wanting = 0
        self.past_119 = 0
                  
        self.current_step = 0
        self.current_episode = 0

        # Step-level tracking (history within episode)
        self.step_history = { 'hunger': [], 'wanting': [],}
    
    def reset(self, **kwargs):
       
        # Reset episode-level trackers
        self.hunger = 0.0
        self.wanting = 0
        self.current_step = 0
        self.past_119 = 0
        
        # Reset step history
        self.step_history = { 'hunger': [], 'wanting': [],}
        
        obs, info = self.env.reset(**kwargs)
        self.current_episode += 1
        
        return obs, info
   
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        self.hunger += self.hunger_inc

        current_119 = int(obs[119])
        
        # If 119 changed == pellet eaten
        if current_119 - self.past_119 == 1:
            self.past_119 = current_119
            self.hunger *= 0.2  # Satiation   
            
        self.hunger = min(self.hunger, self.max_hunger)    
        wanting_reward = -self.lambda_wanting * (self.hunger / self.max_hunger)        

        # Accumulate episode totals
        self.wanting += wanting_reward
        
        # Record step-level data
        self.step_history['hunger'].append(self.hunger)
        self.step_history['wanting'].append(wanting_reward)
        
        self.current_step += 1
        
        if term or trunc:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["want"] = self.wanting
            info["episode"]["step_history"] = self.step_history.copy()
        
        # Keep ONLY this one - CombineRewardWrapper needs it
        info["wanting_reward"] = wanting_reward
        
        return obs, reward, term, trunc, info



# In[ ]:


class HullWrapperOld(gym.Wrapper):

    def __init__(self, env, 
                 lambda_wanting=1.0,    # Drive reduction weight (Hull component)
                 hunger_inc=0.2, 
                 max_hunger=10.0):
        
        super().__init__(env)
        self.lambda_wanting = lambda_wanting
        self.hunger_inc = hunger_inc
        self.max_hunger = max_hunger
        self.hunger = 0.0
        self.wanting = 0
        self.past_119 = 0
                  
        self.current_step = 0
        self.current_episode = 0

        # Step-level tracking (history within episode)
        self.step_history = { 'hunger': [], 'wanting': [],}
    
    def reset(self, **kwargs):
       
        # Reset episode-level trackers
        self.hunger = 0.0
        self.wanting = 0
        self.current_step = 0
        self.past_119 = 0
        
        # Reset step history
        self.step_history = { 'hunger': [], 'wanting': [],}
        
        obs, info = self.env.reset(**kwargs)
        self.current_episode += 1
        
        return obs, info
   
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)

        self.hunger += self.hunger_inc

        current_119 = int(obs[119])
        
        # If 119 changed == pellet eaten
        if current_119 - self.past_119 == 1:
            self.past_119 = current_119
            self.hunger *= 0.2  # Satiation   
            
        self.hunger = min(self.hunger, self.max_hunger)    
        wanting_reward = -self.lambda_wanting * (self.hunger / self.max_hunger)        

        # Accumulate episode totals
        self.wanting += wanting_reward
        
        # Record step-level data
        self.step_history['hunger'].append(self.hunger)
        self.step_history['wanting'].append(wanting_reward)
        
        self.current_step += 1
        
        if term or trunc:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["drive"] = self.D
            info["episode"]["step_history"] = self.step_history.copy()
        
        # Keep ONLY this one - CombineRewardWrapper needs it
        info["wanting_reward"] = wanting_reward
        
        return obs, reward, term, trunc, info


        

