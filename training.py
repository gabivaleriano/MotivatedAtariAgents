#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
import pickle

from tqdm import tqdm
from dqn import DQN
from replay_buffer import ReplayBuffer
from env import make_env_with_metrics
from utils import set_seed

def train_with_seed_incentive(seed=42, 
                              steps=1_000_000,
                              alpha = 0.05,
                              kappa_= 0,
                              loss = 100,
                              like = 1,
                              dqn_modulation = 1,
                              agent = 'Incentive'): 

    if dqn_modulation == 0 or agent != 'Incentive':
        return
        
    set_seed(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env_with_metrics(seed, agent='Incentive', loss=loss, like=like)
    
    # Create networks
    net = DQN(env.action_space.n).to(device)
    target = DQN(env.action_space.n).to(device)
    target.load_state_dict(net.state_dict())

    # Create networks
    cue_net = DQN(env.action_space.n).to(device)
    cue_target = DQN(env.action_space.n).to(device)
    cue_target.load_state_dict(cue_net.state_dict())
    
    # Optimizer and buffer
    #opt = optim.Adam(net.parameters(), lr=6.25e-5, eps=0.01/32)
    opt = optim.RMSprop(net.parameters(), lr=0.00025, alpha=0.95, eps=0.01, momentum=0.0)
    cue_opt = optim.RMSprop(cue_net.parameters(), lr=0.00025, alpha=0.95, eps=0.01, momentum=0.0)
    buf = ReplayBuffer()
    cue_buf = ReplayBuffer()
    #eps = lambda t: 0.1 + 0.9 * np.exp(-t / 500000) 
    eps = lambda t: max(0.1, 1.0 - 0.9 * (t / 1_000_000))
    
    state, _ = env.reset(seed=seed)
    state = np.asarray(state)
    info = {}
    
    episode_count = 0
    
    bar = tqdm(total=steps, desc=f"Seed {seed}")

    all_metrics = []
    
    for t in range(1, steps + 1):
        # Epsilon-greedy action selection
        if random.random() < eps(t):
            a = env.action_space.sample()

        else:
            with torch.no_grad():
                q = net(torch.tensor(state, device=device).unsqueeze(0))
                q_values = q.squeeze(0).cpu().numpy()    

                cue_q = cue_net(torch.tensor(state, device=device).unsqueeze(0))
                cue_q_values = cue_q.squeeze(0).cpu().numpy()  

                # softmax normalization over cue_q_values
                cue_q_values[0] = 0
                remaining = cue_q_values[1:]
                exp_vals = np.exp(remaining - np.max(remaining))  # subtract max for numerical stability
                probs = exp_vals / exp_vals.sum()
                cue_q_values[1:] = probs

                # softmax normalization over q_values
                aux = q_values
                exp_vals = np.exp(aux - np.max(aux))  # subtract max for numerical stability
                probs = exp_vals / exp_vals.sum()
                q_values = probs       
                

            kappa = info.get('kappa', None)
            alpha = alpha
            if kappa_ == 1 and t > 50000 :
                    q_values = q_values * (1 + alpha * kappa_ * cue_q_values)
            elif kappa_ == 2 and t > 50000 :
                    q_values = q_values * (1 + alpha * kappa_ * np.array([0, 0.25, 0.25, 0.25, 0.25]))                
            else:                      
                if kappa is not None and kappa > 0 and t > 50000: 
                    q_values = q_values * (1 + alpha * kappa * cue_q_values)
               
            a = int(np.argmax(q_values)) 
            
        # Environment step
        ns, r, term, trunc, info = env.step(a)
        ns = np.asarray(ns)
        done = term or trunc
        cue_reward = info['R*']
        
        # Store experience
        buf.push(state, a, r, ns, done)
        cue_buf.push(state, a, cue_reward, ns, done)
        state = ns
        bar.update(1)
        
        # Episode ended
        if done:
            episode_count += 1
            state, _ = env.reset(seed=seed)
            state = np.asarray(state)

            if 'metrics' in info:
                metrics = info['metrics']
                metrics['episode'] = episode_count
                metrics['total reward'] = info["episode"].get("r", 0)                   
                
                all_metrics.append(metrics)
        
        # Training step main dqn
        if len(buf) >= 10000 and t % 4 == 0:
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
            loss = F.smooth_l1_loss(q, tgt)  # Huber loss, beta=1.0 by default -> clips TD error to [-1,1]
            opt.zero_grad()
            loss.backward()
            opt.step()

            
        # Update target network
        if t % 10000 == 0:
            target.load_state_dict(net.state_dict())
        ############################Training cue dqn #################################################################

        if len(cue_buf) >= 10000 and t % 4 == 0:
            s, a_batch, r_batch, ns, d = cue_buf.sample(32)
            s = torch.tensor(s, device=device)
            ns = torch.tensor(ns, device=device)
            a_batch = torch.tensor(a_batch, device=device).unsqueeze(1)
            r_batch = torch.tensor(r_batch, device=device).unsqueeze(1)
            d = torch.tensor(d, device=device).unsqueeze(1)
            
            # Compute Q-values
            q = cue_net(s).gather(1, a_batch)
            
            # Compute target
            with torch.no_grad():
                nq = cue_target(ns).max(1)[0].unsqueeze(1)
                tgt = r_batch + 0.99 * nq * (1 - d)
            
            # Update network
            loss = F.smooth_l1_loss(q, tgt)  # Huber loss, beta=1.0 by default -> clips TD error to [-1,1]
            cue_opt.zero_grad()
            loss.backward()
            cue_opt.step()
            
        # Update target network
        if t % 10000 == 0:
            cue_target.load_state_dict(cue_net.state_dict())
        ###########################################################################################################
  
        # Update progress bar
        if t % 50000 == 0:
            #recent = np.mean(rewards_history[-20:]) if rewards_history else 0
            bar.set_postfix({"eps": f"{eps(t):.2f}"})#, "reward": f"{recent:.0f}"})
    
    bar.close()   
       
    return net, cue_net, all_metrics


# In[2]:


def train_with_seed(seed=42, 
                    steps=1_000_000,
                    dqn_modulation=1,
                    loss=100,
                    agent = 'Vanilla'):
    
    if agent == 'Incentive' and dqn_modulation == 1:
        return
    
    set_seed(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env_with_metrics(seed, loss=loss, agent=agent)
    
    # Create networ
    net = DQN(env.action_space.n).to(device)
    target = DQN(env.action_space.n).to(device)
    target.load_state_dict(net.state_dict())
    
    # Optimizer and buffer
    opt = optim.RMSprop(net.parameters(), lr=0.00025, alpha=0.95, eps=0.01, momentum=0.0)
    buf = ReplayBuffer()
    eps = lambda t: max(0.1, 1.0 - 0.9 * (t / 1_000_000))
    
    state,_ = env.reset(seed=seed)
    state = np.asarray(state)
    info = {}
    
    episode_count = 0
    
    bar = tqdm(total=steps, desc=f"Seed {seed}")

    all_metrics = []
    
    for t in range(1, steps + 1):
        # Epsilon-greedy action selection
        if random.random() < eps(t):
            a = env.action_space.sample()

        else:
            with torch.no_grad():
                q = net(torch.tensor(state, device=device).unsqueeze(0))
                q_values = q.squeeze(0).cpu().numpy()    
                
            if agent == 'Incentive':
                
                # softmax normalization over q_values
                aux = q_values
                exp_vals = np.exp(aux - np.max(aux))  # subtract max for numerical stability
                probs = exp_vals / exp_vals.sum()
                q_values = probs  
                
                kappa = info.get('kappa', None)                
                alpha = 0.05
                if kappa is not None and kappa > 0 and t > 50000:
                    C = info.get('C')
                    q_values = q_values * (1 + alpha * kappa * C)
                    
            a = int(np.argmax(q_values)) 
            
        # Environment step
        ns, r, term, trunc, info = env.step(a)
        ram = env.unwrapped.ale.getRAM()
        ns = np.asarray(ns)
        done = term or trunc
        
        # Store experience
        buf.push(state, a, r, ns, done)
        state = ns
        bar.update(1)
        
        # Episode ended
        if done:
            episode_count += 1
            state, _ = env.reset(seed=seed)
            state = np.asarray(state)

            if 'metrics' in info:
                metrics = info['metrics']
                metrics['episode'] = episode_count
                metrics['total reward'] = info["episode"].get("r", 0)

                all_metrics.append(metrics)
                
        # Training step
        if len(buf) >= 10000 and t % 4 == 0:
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
            #loss = F.mse_loss(q, tgt)
            loss = F.smooth_l1_loss(q, tgt)  # Huber loss, beta=1.0 by default -> clips TD error to [-1,1]
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        # Update target network
        if t % 10000 == 0:
            target.load_state_dict(net.state_dict())
        
        # Update progress bar
        if t % 50000 == 0:
            bar.set_postfix({"eps": f"{eps(t):.2f}"})
    
    bar.close()    
    
    return net, all_metrics


# In[3]:


def complete_training(num_seeds=5, 
                   steps=1_000_000,
                   agents=['Vanilla', 'Incentive'],
                   eval_episodes = 100,
                   alpha=0.05,
                   loss=100,
                   kappa_=0,
                   like=1,
                   dqn_modulation=1,
                   save_dir='results'):

    os.makedirs(save_dir, exist_ok=True)
    
    all_results = {}

    # Train each agent 
    for agent in agents:
        print(f"\n{'='*60}")
        print(f"TRAINING AGENT: {agent}")
        print(f"{'='*60}\n")
        
        agent_results = {
            'training': [],
            'evaluation': []
        }
        
        # Train with multiple seeds
        seeds = [123, 456][:num_seeds]

        if agent == 'Vanilla' or agent == 'Hull' or agent == 'WantLike' or (agent == 'Incentive' and dqn_modulation == 0): 
            for seed in seeds: 
                net, metrics = train_with_seed(
                    seed=seed,
                    steps=steps,
                    loss=loss,
                    dqn_modulation = dqn_modulation,
                    agent=agent )
                     
                agent_results['training'].append({
                'seed': seed,
                'metrics': metrics})

                # Evaluate
                eval_metrics = evaluate_agent(
                    net=net,
                    num_episodes=eval_episodes,
                    base_seed=seed * 1000,
                    loss=loss,
                    dqn_modulation = dqn_modulation,
                    agent = agent
                )
                
                agent_results['evaluation'].append({
                    'train_seed': seed,
                    'eval_metrics': eval_metrics
                })
                        
            all_results[agent] = agent_results

        if agent == 'Incentive' and dqn_modulation == 1:
            for seed in seeds:    
                net, cue_net, metrics = train_with_seed_incentive(
                seed=seed,
                steps=steps,
                loss=loss,
                alpha=alpha,
                kappa_=kappa_,
                like=like,
                dqn_modulation = dqn_modulation,
                agent = agent
                )   
                
                agent_results['training'].append({
                    'seed': seed,
                    'metrics': metrics})           

                # Evaluate
                eval_metrics = evaluate_agent_incentive(
                    net=net,
                    cue_net=cue_net,
                    num_episodes=eval_episodes,
                    base_seed=seed * 1000,
                    alpha=alpha,
                    loss=loss,
                    kappa_=kappa_,
                    like=like,
                    dqn_modulation=dqn_modulation,
                    agent=agent
                )
                
                agent_results['evaluation'].append({
                    'train_seed': seed,
                    'eval_metrics': eval_metrics
                })
                
            all_results[agent] = agent_results

    final_path = os.path.join(save_dir, 'results.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(all_results, f)


# In[4]:


def evaluate_agent(net, 
                   num_episodes=100, 
                   base_seed=42, 
                   deterministic=True,
                   dqn_modulation=1,
                   loss=100,
                   agent = 'Vanilla'):

    if agent == 'Incentive' and dqn_modulation == 1:
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\n{'='*60}")
    print(f"Evaluating {agent} agent for {num_episodes} episodes")
    print(f"{'='*60}\n")
    
    env = make_env_with_metrics(base_seed, loss=loss, agent=agent)
    net.eval()
    
    eval_metrics = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluation"):
        # Use different seed for each episode
        episode_seed = base_seed + episode
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        random.seed(episode_seed)
        
        state, _ = env.reset(seed=episode_seed)
        info = {}
        done = False
        
        while not done:
            with torch.no_grad():
                q = net(torch.tensor(state.__array__(), device=device).unsqueeze(0))
                q_values = q.squeeze(0).cpu().numpy()
                
                if agent == 'Incentive':
                    
                    # softmax normalization over q_values
                    aux = q_values
                    exp_vals = np.exp(aux - np.max(aux))  # subtract max for numerical stability
                    probs = exp_vals / exp_vals.sum()
                    q_values = probs  
                    
                    kappa = info.get('kappa', None)                
                    alpha = 0.05
                    if kappa is not None and kappa > 0:
                        C = info.get('C')
                        q_values = q_values * (1 + alpha * kappa * C)   

                if deterministic:
                    a = int(np.argmax(q_values))
                else:
                    # Small epsilon for variation
                    if random.random() < 0.05:
                        a = env.action_space.sample()
                    else:
                        a = int(np.argmax(q_values))
            
            state, r, term, trunc, info = env.step(a)
            done = term or trunc
            
            if done and 'metrics' in info:
                metrics = info['metrics']
                metrics['total reward'] = info["episode"].get("r", 0)
                metrics['eval_episode'] = episode
                metrics['eval_seed'] = episode_seed
                    
                eval_metrics.append(metrics)
    
    net.train()
    return eval_metrics


# In[5]:


def evaluate_agent_incentive(net, cue_net,
                   num_episodes=100, 
                   base_seed=42, 
                   deterministic=True,
                   alpha = 0.05,
                   loss = 100,
                   kappa_ = 0,
                   like=1,
                   dqn_modulation = 1,
                   agent = 'Incentive'):

    if dqn_modulation == 0 or agent != 'Incentive':
        return 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"\n{'='*60}")
    print(f"Evaluating Incentive agent for {num_episodes} episodes")
    print(f"{'='*60}\n")

    
    env = make_env_with_metrics(base_seed, loss=loss, agent=agent, like=like)
    net.eval()
    cue_net.eval()
    
    eval_metrics = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluation"):
        # Use different seed for each episode
        episode_seed = base_seed + episode
        np.random.seed(episode_seed)
        torch.manual_seed(episode_seed)
        random.seed(episode_seed)
        
        state, _ = env.reset(seed=episode_seed)
        info = {}
        done = False
        
        while not done:
            with torch.no_grad():
                q = net(torch.tensor(state.__array__(), device=device).unsqueeze(0))
                q_values = q.squeeze(0).cpu().numpy()

                cue_q = cue_net(torch.tensor(state.__array__(), device=device).unsqueeze(0))
                cue_q_values = cue_q.squeeze(0).cpu().numpy()

                # softmax normalization over cue_q_values
                cue_q_values[0] = 0
                remaining = cue_q_values[1:]
                exp_vals = np.exp(remaining - np.max(remaining))  # subtract max for numerical stability
                probs = exp_vals / exp_vals.sum()
                cue_q_values[1:] = probs

                # softmax normalization over q_values
                aux = q_values
                exp_vals = np.exp(aux - np.max(aux))  # subtract max for numerical stability
                probs = exp_vals / exp_vals.sum()
                q_values = probs

                kappa = info.get('kappa', None)
                alpha = alpha
                if kappa_ == 1:
                    q_values = q_values * (1 + alpha * kappa_ * cue_q_values)
                elif kappa_ == 2 and t > 50000 :
                    q_values = q_values * (1 + alpha * kappa_ * np.array([0, 0.25, 0.25, 0.25, 0.25]))     
                else:
                    if kappa is not None and kappa > 0: 
                        q_values = q_values * (1 + alpha * kappa * cue_q_values)

                if deterministic:
                    a = int(np.argmax(q_values))
                else:
                    # Small epsilon for variation
                    if random.random() < 0.05:
                        a = env.action_space.sample()
                    else:
                        a = int(np.argmax(q_values))
            
            state, r, term, trunc, info = env.step(a)
            done = term or trunc
            
            if done and 'metrics' in info:
                metrics = info['metrics']
                metrics['total reward'] = info["episode"].get("r", 0)
                metrics['eval_episode'] = episode
                metrics['eval_seed'] = episode_seed
                
                eval_metrics.append(metrics)

    cue_net.train()
    net.train()
    return eval_metrics

