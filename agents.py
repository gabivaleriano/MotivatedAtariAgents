#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gymnasium as gym
import numpy as np
import pickle 
from utils import compute_directional_pellet_salience

class HullWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super().__init__(env)
        self.D = 30          # start above homeostasis
        self.D_star = 30     # homeostasis level
        self.D_max = 50
        self.D_min = 0
        
        self.current_episode = 0
        self.past_119 = 0


    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        current_119 = int(ram[119])

        energy_delta = -0.1

        # 1. detect eating first (takes priority)
        if current_119 != self.past_119:
            energy_delta += 1
           
        # 2. update drive
        self.D = np.clip(self.D + energy_delta, self.D_min, self.D_max)        

        # 3. compute intrinsic reward
        if self.D < self.D_star:
            Ri = -(((self.D_star - self.D) / self.D_star) ** 0.5)
        else:
            Ri = (self.D - self.D_star) / self.D_star  # note: no penalty per spec
          
        self.past_119 = current_119
           
        info["drive_reward"] = Ri
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
                
        # Reset episode-level trackers
        self.D = self.D_star
        self.past_119 = 0
        
        obs, info = self.env.reset(**kwargs)        
        return obs, info    


# In[ ]:


class WantLikeWrapper(gym.Wrapper):
    
    def __init__(self, env, raw_tracker=None):
        super().__init__(env)
        self.D = 30          # start at homeostasis
        self.D_star = 30     # homeostasis level
        self.D_max = 50
        self.D_min = 0
        self.past_119 = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        current_119 = int(ram[119])

        energy_delta = -0.1

        # 1. detect eating first (takes priority)
        if current_119 != self.past_119:
            energy_delta += 1 

        self.past_119 = current_119

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
        
        # Keep ONLY this one - CombineRewardWrapper needs it
        info["drive_reward"] = Riw
        info["like_reward"] = Ril
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
       
        # Reset episode-level trackers
        self.D = self.D_star         # start at homeostasis
        self.past_119 = 0
        self.past_lives = 0
       
        obs, info = self.env.reset(**kwargs)      
        return obs, info  


# In[ ]:


class IncentiveWrapper(gym.Wrapper):
# sem tolerância
    
    def __init__(self, env, raw_tracker=None, like = 1):
        super().__init__(env)
        self.raw_tracker = raw_tracker
        self.D = 30          # start at homeostasis
        self.D_star = 30     # homeostasis level
        self.D_max = 50
        self.D_min = 0
        self.prev_pos = (85, 98)
        self.kappa = 1
        self.past_119 = 0
        self.past_lives = 0
        self.like = 1

        with open("traversable_positions.pkl", "rb") as f_trav:
            self.traversable_positions = pickle.load(f_trav)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        current_119 = int(ram[119])
        x = int(ram[10])
        y = int(ram[16])

        # 1. detect eating first (takes priority)
        energy_delta = -0.1

        Ril = 0

        # 1. detect eating first (takes priority)
        if current_119 != self.past_119:
            energy_delta += 1
            Ril = self.like

        self.past_119 = current_119

        # 2. update drive
        old_drive = self.D
        self.D = np.clip(self.D + energy_delta, self.D_min, self.D_max)       

        if self.D < self.D_star:
            self.kappa = 1 + (self.D_star - self.D) / self.D_star  # in [0, 1]
        else:
            self.kappa = 1  # well-fed, no salience amplificatiom            

        Ri = Ril

        eaten = info.get('eaten_pellet_positions', set())        
        C = compute_directional_pellet_salience(x, y, self.traversable_positions, eaten)

        info['C'] = C

        # Combine rewards
        info["like_reward"] = Ril
        info["kappa"] = self.kappa
              
        return obs, reward, terminated, truncated, info


    def reset(self, **kwargs):
       
        # Reset episode-level trackers
        self.D = 30          # start at homeostasis
        self.kappa = 1
        self.past_119 = 0

        obs, info = self.env.reset(**kwargs)
        return obs, info  

