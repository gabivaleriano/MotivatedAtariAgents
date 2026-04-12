#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Custom Gymnasium wrappers for Ms. Pac-Man
"""
import gymnasium as gym
import numpy as np
from utils import compute_directional_pellet_salience
import pickle

class RestrictActionsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 0=noop, 1=up, 2=right, 3=left, 4=down
        self.valid_actions = [0, 1, 2, 3, 4]
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))
    
    def step(self, action):
        # map restricted action to original action space
        real_action = self.valid_actions[action]
        return self.env.step(real_action)
   
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class LifeLossWrapper(gym.Wrapper):
    """Terminate episode when a life is lost (RAM byte 123 tracks lives)."""
    
    LIVES_RAM_BYTE = 123  # 0x7B — starts at 2 for Pac-Man
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._lives = obs[self.LIVES_RAM_BYTE]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_lives = obs[self.LIVES_RAM_BYTE]
        
        if current_lives < self._lives:
            reward -= 10
            #terminated = True  
        
        self._lives = current_lives
        return obs, reward, terminated, truncated, info

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

        if self.episode_raw_rewards:
            self.episode_history['extrinsic_total'].append(self.episode_raw_rewards)
        self.episode_raw_rewards = []
        self.cumulative_raw_reward = 0
        self.last_raw_reward = 0
            
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
            #info['raw_rewards_list'] = self.episode_raw_rewards.copy()

        return obs, reward, terminated, truncated, info

class MetricsWrapper(gym.Wrapper):
    """Wrapper that tracks RAM state and calculates metrics"""
    
    def __init__(self, env, raw_tracker=None):
        super().__init__(env)
        self.raw_tracker = raw_tracker
        
        # Episode tracking
        self.episode_positions = []
        self.episode_steps = 0
        self.pellets_eaten = 0
        
        # Level tracking
        self.current_episode_level = 0          # resets each episode, max 8
        self.level_started = False
        self.past_119 = 0    
        self.total_levels_completed = 0

        # Score tracking
        self.past_120 = 0
        self.past_121 = 0
        self.power_pellets_eaten = 0
        self.ghosts_eaten = 0
        
    def reset(self, **kwargs):
        """Reset episode tracking"""
        obs, info = self.env.reset(**kwargs)        
        self.episode_positions = []
        self.episode_steps = 0
        self.pellets_eaten = 0      # add this
        self.past_119 = int(obs[119])  
        self.current_episode_level = 0
        self.level_started = False

        self.past_120 = int(obs[120])
        self.past_121 = int(obs[121])
        self.power_pellets_eaten = 0
        self.ghosts_eaten = 0
               
        return obs, info
    
    def step(self, action):
        """Step environment and track metrics"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Store position (from RAM)
        x = int(obs[10])
        y = int(obs[16])
        self.episode_positions.append((x, y))
        
        self.episode_steps += 1

        current_119 = int(obs[119])

        if current_119 > 0:
            self.level_started = True

        # level transition: byte resets to 0 AFTER pellets were seen
        if current_119 == 0 and self.level_started:
            self.total_levels_completed += 1  # just keep counting, useful metric
            self.current_episode_level = self.total_levels_completed % 8  # 0-7 cycling
            self.level_started = False  # wait for next level's pellets

        if current_119 > self.past_119:
            self.pellets_eaten += 1   
            
        self.past_119 = current_119

        current_120 = int(obs[120])
        current_121 = int(obs[121])
        
        score_now  = (current_121 * 160 + current_120) // 16 * 10
        score_prev = (self.past_121 * 160 + self.past_120) // 16 * 10
        delta = score_now - score_prev
        
        if delta == 50:
            self.power_pellets_eaten += 1
            
        elif delta in [200, 400, 800, 1600]:
            self.ghosts_eaten += 1
            
        self.past_120 = current_120
        self.past_121 = current_121
        
        # Calculate metrics at episode end
        if terminated or truncated:
            raw_rewards = self.raw_tracker.episode_raw_rewards.copy() if self.raw_tracker else []
            metrics = self.calculate_metrics()
            metrics['raw_rewards_list'] = raw_rewards               
            metrics['raw_episode_return'] = sum(raw_rewards) if raw_rewards else 0  
            info['metrics'] = metrics
        
        return obs, reward, terminated, truncated, info       


    def calculate_metrics(self):
        """Calculate all metrics for the episode"""
       
        # 1. Average Lifetime
        lifetime = self.episode_steps
        
        # 2. Pellet Efficiency (from RAW rewards)
        #pellets_eaten = self.pellets_eaten
        #pellet_efficiency = pellets_eaten / lifetime if lifetime > 0 else 0
       
        #ghost_efficiency = self.ghosts_eaten / self.power_pellets_eaten if self.power_pellets_eaten > 0 else 0
        
        # 4. Backtracking Rate
        backtrack_count = 0
        visited_positions = set()
        
        for pos in self.episode_positions:
            if pos in visited_positions:
                backtrack_count += 1
            visited_positions.add(pos)
        
        backtrack_rate = backtrack_count / lifetime if lifetime > 0 else 0
        
        # 5. External (raw) reward
        #external_reward = sum(raw_rewards) if raw_rewards else 0
               
        return {
            'lifetime': lifetime,
            #'pellet_efficiency': pellet_efficiency,
            #'ghost_eating_efficiency': ghost_efficiency,
            'backtracking_rate': backtrack_rate,
            'level_reached': self.total_levels_completed,
            'pellets_eaten': self.pellets_eaten,
            'power_pellets_eaten': self.power_pellets_eaten,
            'ghosts_eaten': self.ghosts_eaten,
        }        


# In[1]:


class VanillaPositionWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)        
        self.step_history = {'C': [], 'x_position': [], 'y_position': [], 'transformed_reward': []} 
        self.eaten_pellet_positions = set()

        with open("traversable_positions.pkl", "rb") as f_trav:
            self.traversable_positions = pickle.load(f_trav)

        low = np.append(self.observation_space.low, [-np.inf] * 5).astype(np.float32)
        high = np.append(self.observation_space.high, [np.inf] * 5).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_position = int(obs[10])
        y_position = int(obs[16])

        curr_pos = (x_position, y_position)

        self.eaten_pellet_positions.add(curr_pos)

        #################################################
        eaten = info.get('eaten_pellet_positions', set())
        
        C = compute_directional_pellet_salience(x_position, y_position, self.traversable_positions, eaten)
        obs = np.append(obs, C)

        #################################################

        self.step_history['x_position'].append(x_position)
        self.step_history['y_position'].append(y_position)
        self.step_history['transformed_reward'].append(reward) 
        
        if terminated or truncated:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["step_history"] = self.step_history.copy()

        info["eaten_pellet_positions"] = self.eaten_pellet_positions
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.step_history = {'C': [], 'x_position': [], 'y_position': [], 'transformed_reward': []}      
        self.eaten_pellet_positions = set()
        
        obs, info = self.env.reset(**kwargs)   
        obs = np.append(obs, [0,0,0,0,0,])
        return obs, info    


# In[ ]:


class HullWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.D = 30          # start above homeostasis
        self.D_star = 30     # homeostasis level
        self.D_max = 50
        self.D_min = 0
        
        self.current_episode = 0
        self.step_history = {'C': [], 'Ri': [], 'x_position': [], 'y_position': [], 'transformed_reward': []}  # ← add Ri
        self.episode_intrinsic_total = 0.0
        self.past_119 = 0
        self.eaten_pellet_positions = set()
        self.past_lives = 0 # not start penalizing

        with open("traversable_positions.pkl", "rb") as f_trav:
            self.traversable_positions = pickle.load(f_trav)

        low = np.append(self.observation_space.low, [-np.inf] * 5).astype(np.float32)
        high = np.append(self.observation_space.high, [np.inf] * 5).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_119 = int(obs[119])
        x_position = int(obs[10])
        y_position = int(obs[16])
        current_lives = int(obs[123])
        curr_pos = (x_position, y_position)

        energy_delta = -0.1

        # 1. detect eating first (takes priority)
        if current_119 != self.past_119:
            energy_delta = +1

        if  (self.past_lives == 2 or self.past_lives == 1) and (self.past_lives - current_lives == 1):
            #energy_delta -= 5
            self.D = self.D_min

        # 2. update drive
        self.D = np.clip(self.D + energy_delta, self.D_min, self.D_max)        

        # 3. compute intrinsic reward
        if self.D < self.D_star:
            Ri = -(((self.D_star - self.D) / self.D_star) ** 0.5)
        else:
            Ri = (self.D - self.D_star) / self.D_star  # note: no penalty per spec

        self.past_lives = current_lives

        #################################################
        eaten = info.get('eaten_pellet_positions', set())
        
        C = compute_directional_pellet_salience(x_position, y_position, self.traversable_positions, eaten)
        obs = np.append(obs, C)

        #################################################


        self.episode_intrinsic_total += Ri            # ← accumulate
        self.step_history['C'].append(C)
        self.step_history['drive'].append(self.D)
        self.step_history['Ri'].append(Ri) 
        self.step_history['x_position'].append(x_position)
        self.step_history['y_position'].append(y_position)
        self.step_history['transformed_reward'].append(reward) 
        self.past_119 = current_119
        self.eaten_pellet_positions.add(curr_pos)

        if terminated or truncated:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["step_history"] = self.step_history.copy()
            info["episode"]["intrinsic_total"] = self.episode_intrinsic_total
        
        # Keep ONLY this one - CombineRewardWrapper needs it
        info["drive_reward"] = Ri
        info["eaten_pellet_positions"] = self.eaten_pellet_positions
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
       
        # Reset episode-level trackers
        self.D = self.D_star          

        self.step_history = {'C': [],'drive': [], 'Ri': [], 'x_position': [], 'y_position': [], 'transformed_reward': []}   # ← reset both
        self.episode_intrinsic_total = 0.0  
        self.past_119 = 0
        self.eaten_pellet_positions = set()
        self.past_lives = 2
        
        obs, info = self.env.reset(**kwargs)
        self.current_episode += 1
        obs = np.append(obs, [0,0,0,0,0,])
        
        return obs, info    
 
 


# In[ ]:


class WantLikeWrapper(gym.Wrapper):
    
    def __init__(self, env, raw_tracker=None):
        super().__init__(env)
        self.D = 30          # start at homeostasis
        self.D_star = 30     # homeostasis level
        self.D_max = 50
        self.D_min = 0

        # Step-level tracking (history within episode)
        self.current_episode = 0
        self.step_history = {'C': [], 'drive': [], 'Riw': [], 'Ril': [], 'x_position': [], 'y_position': [], 'transformed_reward': []}          
        self.episode_intrinsic_total = 0.0
        self.past_119 = 0
        self.eaten_pellet_positions = set()
        self.past_lives = 0 # not start penalizing

        with open("traversable_positions.pkl", "rb") as f_trav:
            self.traversable_positions = pickle.load(f_trav)

        low = np.append(self.observation_space.low, [-np.inf] * 5).astype(np.float32)
        high = np.append(self.observation_space.high, [np.inf] * 5).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        current_119 = int(obs[119])
        x_position = int(obs[10])
        y_position = int(obs[16])

        current_lives = int(obs[123])
        curr_pos = (x_position, y_position)

        energy_delta = -0.1

        # 1. detect eating first (takes priority)
        if current_119 != self.past_119:
            energy_delta = +1 

        self.past_119 = current_119

        if  (self.past_lives == 2 or self.past_lives == 1) and (self.past_lives - current_lives == 1):
            energy_delta = -5

        # 2. update drive
        old_drive = self.D
        self.D = np.clip(self.D + energy_delta, self.D_min, self.D_max)
          
        # without desliking for being above homeostase  

        # if it is under homeosthasis (old drive < 30) compute in both directions, like for increasing, dislike decreasing
        if old_drive <= self.D_star: # if it was under homeostasis
            Ril = (self.D - old_drive)/self.D_star # positive if D increased and negative otherwise

        # still like eating, but reducing... does not deslike 
        else: 
            if self.D > old_drive: 
                Ril = (self.D - old_drive)/(self.D_star + self.D)
            else: Ril = 0 #there is not like or dislike 
 
        # 3. compute intrinsic reward
        if self.D < self.D_star:
            Riw = -(((self.D_star - self.D) / self.D_star) ** 0.5)
        else:
            Riw = (self.D - self.D_star) / self.D_star  # note: no penalty per spec

        Ri = Riw + Ril

        self.past_lives = current_lives

        #################################################
        eaten = info.get('eaten_pellet_positions', set())
        
        C = compute_directional_pellet_salience(x_position, y_position, self.traversable_positions, eaten)
        obs = np.append(obs, C)

        #################################################

        self.episode_intrinsic_total += Ri
        
        self.step_history['C'].append(C)
        self.step_history['drive'].append(self.D)
        self.step_history['Riw'].append(Riw)
        self.step_history['Ril'].append(Ril)
        self.step_history['x_position'].append(x_position)
        self.step_history['y_position'].append(y_position)
        self.step_history['transformed_reward'].append(reward) 
        self.eaten_pellet_positions.add(curr_pos)

        if terminated or truncated:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["step_history"] = self.step_history.copy()
            info["episode"]["intrinsic_total"] = self.episode_intrinsic_total
        
        # Keep ONLY this one - CombineRewardWrapper needs it
        info["want_reward"] = Riw
        info["like_reward"] = Ril
        info["eaten_pellet_positions"] = self.eaten_pellet_positions
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
       
        # Reset episode-level trackers
        self.D = self.D_star         # start at homeostasis
        self.past_119 = 0

        # Step-level tracking (history within episode)
        self.step_history = {'C': [], 'drive': [], 'Riw': [], 'Ril': [], 'x_position': [], 'y_position': [], 'transformed_reward': []} 
        self.episode_intrinsic_total = 0.0
        
        obs, info = self.env.reset(**kwargs)
        obs = np.append(obs, [0,0,0,0,0,])
        self.current_episode += 1        
        
        return obs, info     

class CombineRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # collect whatever intrinsic rewards are present
        intrinsic_keys = ["drive_reward", "want_reward", "like_reward"]
        intrinsic = sum(info.get(k, 0.0) for k in intrinsic_keys)
        
        total = reward + intrinsic
        
        info["extrinsic_reward"] = reward
        info["intrinsic_reward"] = intrinsic
        info["total_reward"] = total
        
        return obs, total, term, trunc, info


# In[ ]:


class IncentiveWrapper(gym.Wrapper):
# sem tolerância
    
    def __init__(self, env, raw_tracker=None):
        super().__init__(env)
        self.raw_tracker = raw_tracker
        self.D = 30          # start at homeostasis
        self.D_star = 30     # homeostasis level
        self.D_max = 50
        self.D_min = 0
        self.prev_pos = (85, 98)
        self.kappa = 1
        self.eaten_pellet_positions = set()
        self.past_119 = 0
        # track everywhere Pac-Man has actually been
        self.traversable_positions = set()
        self.episode_intrinsic_total = 0
        

        # Step-level tracking (history within episode)
        self.current_episode = 0
        self.step_history = {'C': [], 'drive': [], 'kappa': [], 'Ril': [],  'x_position': [], 'y_position': [], 'transformed_reward': []}

        with open("traversable_positions.pkl", "rb") as f_trav:
            self.traversable_positions = pickle.load(f_trav)

        low = np.append(self.observation_space.low, [-np.inf] * 5).astype(np.float32)
        high = np.append(self.observation_space.high, [np.inf] * 5).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_position = int(obs[10])
        y_position = int(obs[16])

        curr_pos = (x_position, y_position)   # x_byte=10, y_byte=16
        
        if curr_pos not in self.traversable_positions:
            self.traversable_positions.add(curr_pos)

        # 1. detect eating first (takes priority)
        energy_delta = -0.1

        current_119 = int(obs[119])

        Ril = 0

        # 1. detect eating first (takes priority)
        if current_119 != self.past_119:
            energy_delta = +1
            Ril = 1
            
        self.eaten_pellet_positions.add(curr_pos)

        self.past_119 = current_119

        # 2. update drive
        old_drive = self.D
        self.D = np.clip(self.D + energy_delta, self.D_min, self.D_max)       

        if self.D < self.D_star:
            self.kappa = 1 + (self.D_star - self.D) / self.D_star  # in [0, 1]
        else:
            self.kappa = 1  # well-fed, no salience amplificatiom            

        Ri = Ril

        #################################################
        eaten = info.get('eaten_pellet_positions', set())
        
        C = compute_directional_pellet_salience(x_position, y_position, self.traversable_positions, eaten)
        obs = np.append(obs, C)

        #################################################

        self.episode_intrinsic_total += Ri        
        self.step_history['drive'].append(self.D)
        self.step_history['C'].append(C)
        self.step_history['kappa'].append(self.kappa)
        self.step_history['Ril'].append(Ril)
        self.step_history['x_position'].append(x_position)
        self.step_history['y_position'].append(y_position)
        self.step_history['transformed_reward'].append(reward) 


        if terminated or truncated:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["step_history"] = self.step_history.copy()
            info["episode"]["intrinsic_total"] = self.episode_intrinsic_total

        info["eaten_pellet_positions"] = self.eaten_pellet_positions
        info["traversable_positions"] = self.traversable_positions
        info["pacman_pos"] = curr_pos

        # Combine rewards
        info["like_reward"] = Ril
        info["kappa"] = self.kappa
        info["C"] = C
               
        return obs, reward, terminated, truncated, info


    def reset(self, **kwargs):
       
        # Reset episode-level trackers
        self.D = 30          # start at homeostasis
        self.kappa = 1
        self.eaten_pellet_positions = set()
        self.past_119 = 0

        # Step-level tracking (history within episode)
        self.step_history = {'C': [], 'drive': [], 'kappa': [], 'Ril': [],  'x_position': [], 'y_position': [], 'transformed_reward': []} 
        
        obs, info = self.env.reset(**kwargs)
        self.current_episode += 1
        self.episode_intrinsic_total = 0

        obs = np.append(obs, [0,0,0,0,0,])
        
        return obs, info  


# In[ ]:


class OldHullWrapper(gym.Wrapper):

    def __init__(self, env, 
                 lambda_wanting=1.0,    
                 hunger_inc=0.2, 
                 max_hunger=10.0):
        
        super().__init__(env)
        self.lambda_wanting = lambda_wanting
        self.hunger_inc = hunger_inc
        self.max_hunger = max_hunger
        self.hunger = 0.0
        self.wanting = 0
        self.past_119 = 0

        #x_position = int(obs[10])
        #y_position = int(obs[16])
                  
        self.current_step = 0
        self.current_episode = 0

        # Step-level tracking (history within episode)
        self.step_history = {'drive': [], 'kappa': [], 'Ril': [],  'x_position': [], 'y_position': [], 'transformed_reward': []} 
    
    def reset(self, **kwargs):
       
        # Reset episode-level trackers
        self.hunger = 0.0
        self.wanting = 0
        self.current_step = 0
        self.past_119 = 0
        
        # Reset step history
        self.step_history = {'drive': [], 'kappa': [], 'Ril': [],  'x_position': [], 'y_position': [], 'transformed_reward': []} 
        
        obs, info = self.env.reset(**kwargs)
        self.current_episode += 1
        
        return obs, info
   
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        x_position = int(obs[10])
        y_position = int(obs[16])

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
        self.step_history['drive'].append(self.hunger)
        self.step_history['Ril'].append(wanting_reward)
        self.step_history['x_position'].append(x_position)
        self.step_history['y_position'].append(y_position)
        self.step_history['transformed_reward'].append(reward) 
        
        self.current_step += 1
        
        if term or trunc:
            if "episode" not in info:
                info["episode"] = {}
            info["episode"]["drive"] = self.hunger
            info["episode"]["step_history"] = self.step_history.copy()
        
        # Keep ONLY this one - CombineRewardWrapper needs it
        info["want_reward"] = wanting_reward
        
        return obs, reward, term, trunc, info


        


# In[ ]:




