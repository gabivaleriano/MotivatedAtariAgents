#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Evaluation functions for trained DQN agents
"""
import torch
import numpy as np
import random
from tqdm import tqdm

from environment import make_env_with_metrics


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_agent(net, env_name, num_episodes=100, base_seed=42, deterministic=True):
    """Evaluate trained agent over multiple episodes with different seeds"""
    print(f"\n{'='*60}")
    print(f"Evaluating agent for {num_episodes} episodes")
    print(f"{'='*60}\n")
    
    env = make_env_with_metrics(env_name, use_ram=True)
    net.eval()
    
    eval_metrics = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluation"):
        # Use different seed for each episode
        episode_seed = base_seed + episode
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        random.seed(episode_seed)
        
        state, _ = env.reset(seed=episode_seed)
        done = False
        
        while not done:
            with torch.no_grad():
                q = net(torch.tensor(state.__array__(), device=device).unsqueeze(0))
                if deterministic:
                    a = q.argmax(1).item()
                else:
                    # Small epsilon for variation
                    if random.random() < 0.05:
                        a = env.action_space.sample()
                    else:
                        a = q.argmax(1).item()
            
            state, r, term, trunc, info = env.step(a)
            done = term or trunc
            
            if done and 'metrics' in info:
                metrics = info['metrics']
                metrics['eval_episode'] = episode
                metrics['eval_seed'] = episode_seed
                eval_metrics.append(metrics)
    
    return eval_metrics

