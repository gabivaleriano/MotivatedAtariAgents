#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gymnasium as gym

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


# In[ ]:


class MetricsWrapper(gym.Wrapper):
    """Wrapper that tracks RAM state and calculates metrics"""
    
    def __init__(self, env, raw_tracker=None):
        super().__init__(env)
        
        # Episode tracking
        self.episode_positions = []
        self.episode_steps = 0
        self.pellets_eaten = 0
        
        # Level tracking
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
        ram = self.env.unwrapped.ale.getRAM()
        
        obs, info = self.env.reset(**kwargs)        
        self.episode_positions = []
        self.episode_steps = 0
        self.pellets_eaten = 0     
        self.past_119 = int(ram[119])  
        self.current_episode_level = 0
        self.level_started = False

        self.past_120 = int(ram[120])
        self.past_121 = int(ram[121])
        self.power_pellets_eaten = 0
        self.ghosts_eaten = 0
               
        return obs, info
    
    def step(self, action):
        """Step environment and track metrics"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        ram = self.env.unwrapped.ale.getRAM()
        
        # Store position (from RAM)
        x = int(ram[10])
        y = int(ram[16])
        self.episode_positions.append((x, y))

        # Increase step
        self.episode_steps += 1

        # Identify current 119 - pellets counter
        current_119 = int(ram[119])
        if current_119 > 0:
            self.level_started = True

        # level transition: byte resets to 0 AFTER pellets were seen
        if current_119 == 0 and self.level_started:
            self.total_levels_completed += 1  # just keep counting
            self.level_started = False  # wait for next level's pellets

        if current_119 > self.past_119:
            self.pellets_eaten += 1   
            
        self.past_119 = current_119

        current_120 = int(ram[120])
        current_121 = int(ram[121])
        
        score_now  = (current_121 * 160 + current_120) // 16 * 10
        score_prev = (self.past_121 * 160 + self.past_120) // 16 * 10
        delta = score_now - score_prev
        
        if delta == 50:     
            self.power_pellets_eaten += 1
            
        elif delta in [200, 400, 800, 1600]:
            self.ghosts_eaten += 1
            
        self.past_120 = current_120
        self.past_121 = current_121

        info['R*'] = self.pellets_eaten
        
        # Calculate metrics at episode end
        if terminated or truncated:            
            metrics = self.calculate_metrics()
            #metrics['reward'] = reward  
            info['metrics'] = metrics
        
        return obs, reward, terminated, truncated, info       


    def calculate_metrics(self):
        """Calculate all metrics for the episode"""
       
        # 4. Backtracking Rate
        backtrack_count = 0
        visited_positions = set()
        
        for pos in self.episode_positions:
            if pos in visited_positions:
                backtrack_count += 1
            visited_positions.add(pos)
        
        backtrack_rate = backtrack_count / self.episode_steps if self.episode_steps > 0 else 0
               
        return {
            'lifetime': self.episode_steps,
            'backtracking_rate': backtrack_rate,
            'level_reached': self.total_levels_completed,
            'pellets_eaten': self.pellets_eaten,
            'power_pellets_eaten': self.power_pellets_eaten,
            'ghosts_eaten': self.ghosts_eaten,
        }     


# In[ ]:


def scale_reward(r):
    """Transform rewards to reasonable range"""
    if r <= 0:
        return r # negative reward
    elif r >= 10:
        return r / 10 

def apply_reward_shaping(env):
    """Apply reward shaping to environment"""
    return TransformReward(env, scale_reward)

