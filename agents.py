#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gymnasium as gym
import numpy as np

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
        self.current_episode += 1
        
        return obs, info    


# In[ ]:




