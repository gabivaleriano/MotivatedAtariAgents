#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Training functions for DQN on Ms. Pac-Man
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
import pickle
from datetime import datetime
from tqdm import tqdm

from dqn import DQN_RAM
from environment import make_env_with_metrics
from utils import ReplayBuffer, set_seed


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_with_seed(env_name, seed, total_steps=1_000_000, save_dir='results'):
    """Train one agent with a specific seed"""
    print(f"\n{'='*60}")
    print(f"Training with seed {seed}")
    print(f"{'='*60}\n")
    
    # Set seed using utils function
    set_seed(seed)
    
    # Create environment
    env = make_env_with_metrics(env_name, use_ram=True)
    
    # Create networks
    net = DQN_RAM(env.action_space.n).to(device)
    target = DQN_RAM(env.action_space.n).to(device)
    target.load_state_dict(net.state_dict())
    
    # Optimizer and buffer
    opt = optim.Adam(net.parameters(), lr=6.25e-5, eps=0.01/32)
    buf = ReplayBuffer()
    eps = lambda t: 0.1 + 0.9 * np.exp(-t / 500000)
    
    state, _ = env.reset()
    
    # Metrics storage
    all_metrics = []
    rewards_history = []
    episode_count = 0
    
    bar = tqdm(total=total_steps, desc=f"Seed {seed}")
    
    for t in range(1, total_steps + 1):
        # Epsilon-greedy action selection
        if random.random() < eps(t):
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                q = net(torch.tensor(state.__array__(), device=device).unsqueeze(0))
                a = q.argmax(1).item()
        
        # Environment step
        ns, r, term, trunc, info = env.step(a)
        done = term or trunc
        
        # Store experience
        buf.push(state.__array__(), a, r, ns.__array__(), done)
        state = ns
        bar.update(1)
        
        # Episode ended
        if done:
            rewards_history.append(info["episode"]["r"])
            
            # Store metrics
            if 'metrics' in info:
                metrics = info['metrics']
                metrics['episode'] = episode_count
                metrics['timestep'] = t
                metrics['seed'] = seed
                all_metrics.append(metrics)
            
            episode_count += 1
            state, _ = env.reset()
        
        # Training step
        if len(buf) >= 32:
            s, a_batch, r_batch, ns, d = buf.sample(32)
            s = torch.tensor(s, device=device)
            ns = torch.tensor(ns, device=device)
            a_batch = torch.tensor(a_batch, device=device).unsqueeze(1)
            r_batch = torch.tensor(r_batch, device=device).unsqueeze(1)
            d = torch.tensor(d, device=device).unsqueeze(1)
            
            # Compute Q-values
            q = net(s).gather(1, a_batch)
            
            # Compute target
            with torch.no_grad():
                nq = target(ns).max(1)[0].unsqueeze(1)
                tgt = r_batch + 0.99 * nq * (1 - d)
            
            # Update network
            loss = F.mse_loss(q, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Update target network
        if t % 10000 == 0:
            target.load_state_dict(net.state_dict())
        
        # Update progress bar
        if t % 50000 == 0:
            recent = np.mean(rewards_history[-20:]) if rewards_history else 0
            
            if len(all_metrics) >= 20:
                recent_metrics = all_metrics[-20:]
                avg_lifetime = np.mean([m['lifetime'] for m in recent_metrics])
                avg_pellet_eff = np.mean([m['pellet_efficiency'] for m in recent_metrics])
                
                bar.set_postfix({
                    "eps": f"{eps(t):.2f}",
                    "reward": f"{recent:.0f}",
                    "life": f"{avg_lifetime:.0f}",
                    "p_eff": f"{avg_pellet_eff:.3f}"
                })
            else:
                bar.set_postfix({"eps": f"{eps(t):.2f}", "reward": f"{recent:.0f}"})
    
    bar.close()
    
    # Save results
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'seed': seed,
        'net_state_dict': net.state_dict(),
        'rewards_history': rewards_history,
        'all_metrics': all_metrics,
        'total_steps': total_steps,
        'timestamp': datetime.now().isoformat()
    }
    
    save_path = os.path.join(save_dir, f'training_seed_{seed}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nSaved training results to {save_path}")
    
    return net, rewards_history, all_metrics


def run_full_experiment(env_name="MsPacman", 
                       num_seeds=5, 
                       training_steps=1_000_000,
                       eval_episodes=100,
                       save_dir='results'):
    """Run complete experiment: multiple training runs + evaluation"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {
        'training': [],
        'evaluation': []
    }
    
    seeds = [1, 42, 123, 456, 789][:num_seeds]
    
    for seed in seeds:
        # Train
        net, rewards, metrics = train_with_seed(
            env_name, 
            seed=seed, 
            total_steps=training_steps,
            save_dir=save_dir
        )
        
        all_results['training'].append({
            'seed': seed,
            'rewards': rewards,
            'metrics': metrics
        })
        
        # Evaluate
        from evaluate import evaluate_agent
        
        eval_metrics = evaluate_agent(
            net, 
            env_name, 
            num_episodes=eval_episodes,
            base_seed=seed * 1000,
            deterministic=True
        )
        
        all_results['evaluation'].append({
            'train_seed': seed,
            'eval_metrics': eval_metrics
        })
        
        # Save evaluation
        eval_save_path = os.path.join(save_dir, f'evaluation_seed_{seed}.pkl')
        with open(eval_save_path, 'wb') as f:
            pickle.dump(eval_metrics, f)
    
    # Save combined results
    final_save_path = os.path.join(save_dir, 'all_results.pkl')
    with open(final_save_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\nExperiment complete! Results saved to {save_dir}")
    
    return all_results


# In[ ]:




