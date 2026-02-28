#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Main entry point for Ms. Pac-Man DQN experiments
Run training with multiple seeds and evaluation
"""
import argparse
import os
from experiment import train_with_seed, run_full_experiment, evaluate_agent
from utils import set_seed


def main():
    """Main function to run experiments"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DQN on Ms. Pac-Man')
    
    # Experiment settings
    parser.add_argument('--env', type=str, default='MsPacman',
                       help='Environment name')
    parser.add_argument('--num_seeds', type=int, default=1,
                       help='Number of seeds for training')
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Total training steps per seed')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--agent_styles', type=str, nargs ='+',  default= ['Vanilla', 'Hull'],
                       choices = ['Vanilla', 'Hull', 'Want_like', 'Incentive']
                       help='List with agents to be trained. Options: Vanilla, Hull, Want_like, Incentive')

    parser.add_argument('--agent-styles', type=str, 
                       default=['Vanilla'],
                       choices=['Vanilla', 'Hull', 'Want_like', 'Incentive'],
                       help='List of agent styles to train. Options: Vanilla, Hull, Want_like, Incentive')
    
    # Output settings
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*60)
    print("MS. PAC-MAN DQN EXPERIMENT")
    print("="*60)
    print(f"Environment: {args.env}")
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Training steps: {args.steps:,}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Agent styles: {args.agent_styles}")
    print(f"Save directory: {args.save_dir}")
    print("="*60)
    print()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)    
       
    print("\nRunning full experiment...")
    run_full_experiment(
        env_name=args.env,
        num_seeds=args.num_seeds,
        training_steps=args.steps,
        eval_episodes=args.eval_episodes,
        save_dir=args.save_dir,
        agent_styles=args.agent_styles,
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {args.save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


# In[ ]:


"""
Examples

# Full experiment with default settings (5 seeds, 1M steps each)
python main.py --mode full

# Train single agent
python main.py --mode train --num_seeds 1 --steps 1000

# Full experiment with custom settings
python main.py --mode full --num_seeds 3 --steps 2000000 --eval-episodes 50

# Use reward clipping instead of shaping
python main.py --mode full --reward-shaping clip

# Save to custom directory
python main.py --mode full --save-dir my_experiment_results

# Quick test run (fewer steps)
python main.py --mode train --seeds 1 --steps 10000

#python main.py --num-seeds 3 --steps 500000 --agent-styles Vanilla Hull

"""

