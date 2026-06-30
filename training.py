#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

from tqdm import tqdm
from dqn import DQN
from replay_buffer import ReplayBuffer
from env import make_env_with_metrics
from utils import set_seed



def train_with_seed_incentive(seed=42, 
                    total_steps=1_000_000):  
    
    set_seed(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env_with_metrics(seed)
    
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
    
    bar = tqdm(total=total_steps, desc=f"Seed {seed}")

    all_metrics = []
    #episode_q_before = []   # q values before kappa adjustment
    #episode_cue_q = []
    #episode_q_after  = []   # q values after kappa adjustment  
    
    for t in range(1, total_steps + 1):
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
                
            #if agent_style == 'Incentive':
                #kappa = info.get('kappa', None)
            kappa = 1
            alpha = 0.05
            if kappa is not None and kappa > 0 and t > 50000: #50000
                episode_q_before.append(q_values.copy())
                episode_cue_q.append(cue_q_values.copy())
                q_values = q_values * (1 + alpha * kappa * cue_q_values)

                episode_q_after.append(q_values.copy())

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
                #metrics['step_history'] = info["episode"].get("step_history", {})
                metrics['total reward'] = info["episode"].get("r", 0)

                #metrics['q_before']  = episode_q_before.copy()
                #metrics['q_after']   = episode_q_after.copy()
                #metrics['cue_q']  = episode_cue_q.copy()
                    

                all_metrics.append(metrics)
        

        #episode_q_before.clear()
        #episode_q_after.clear()
        
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
            #loss = F.mse_loss(q, tgt)
            loss = F.smooth_l1_loss(q, tgt)  # Huber loss, beta=1.0 by default -> clips TD error to [-1,1]
            opt.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            opt.step()
            # Huber loss already bounds the gradient contribution per-sample at the error level
            
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
            #loss = F.mse_loss(q, tgt)
            loss = F.smooth_l1_loss(q, tgt)  # Huber loss, beta=1.0 by default -> clips TD error to [-1,1]
            cue_opt.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            cue_opt.step()
            # Huber loss already bounds the gradient contribution per-sample at the error level
            
        # Update target network
        if t % 10000 == 0:
            cue_target.load_state_dict(cue_net.state_dict())
        ###########################################################################################################
  
        # Update progress bar
        if t % 50000 == 0:
            #recent = np.mean(rewards_history[-20:]) if rewards_history else 0
            bar.set_postfix({"eps": f"{eps(t):.2f}"})#, "reward": f"{recent:.0f}"})
    
    bar.close()   

    final_path = os.path.join(save_dir, 'results_incentive.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(all_results, f)
        
    return all_metrics


# In[13]:


def train_with_seed(seed=42, 
                    total_steps=1_000_000,
                    save_dir = 'results',
                    incentive = False):

    if incentive = True:
        train_with_seed_incentive(seed = seed, total_steps=total_steps, save_dir = save_dir)
        return
    
    set_seed(seed=seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment
    env = make_env_with_metrics(seed)
    
    # Create networks
    net = DQN(env.action_space.n).to(device)
    target = DQN(env.action_space.n).to(device)
    target.load_state_dict(net.state_dict())
    
    # Optimizer and buffer
    #opt = optim.Adam(net.parameters(), lr=6.25e-5, eps=0.01/32)
    opt = optim.RMSprop(net.parameters(), lr=0.00025, alpha=0.95, eps=0.01, momentum=0.0)
    buf = ReplayBuffer()
    #eps = lambda t: 0.1 + 0.9 * np.exp(-t / 500000) 
    eps = lambda t: max(0.1, 1.0 - 0.9 * (t / 1_000_000))
    
    state,_ = env.reset(seed=seed)
    state = np.asarray(state)
    info = {}
    
    episode_count = 0
    
    bar = tqdm(total=total_steps, desc=f"Seed {seed}")

    all_metrics = []
    
    for t in range(1, total_steps + 1):
        # Epsilon-greedy action selection
        if random.random() < eps(t):
            a = env.action_space.sample()

        else:
            with torch.no_grad():
                q = net(torch.tensor(state, device=device).unsqueeze(0))
                q_values = q.squeeze(0).cpu().numpy()    
           
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
                #metrics['step_history'] = info["episode"].get("step_history", {})
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
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10)
            opt.step()
            # Huber loss already bounds the gradient contribution per-sample at the error level
        
        # Update target network
        if t % 10000 == 0:
            target.load_state_dict(net.state_dict())
        
        # Update progress bar
        if t % 50000 == 0:
            bar.set_postfix({"eps": f"{eps(t):.2f}"})
    
    bar.close()    

    final_path = os.path.join(save_dir, 'results.pkl')
    with open(final_path, 'wb') as f:
        pickle.dump(all_results, f)
        
    return all_metrics

