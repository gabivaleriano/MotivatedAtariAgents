#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Main entry point for Ms. Pac-Man DQN experiments
Run training with multiple seeds and evaluation
"""
import argparse
import os
from experiment import train_with_seed, run_full_experiment
from evaluate import evaluate_agent
from utils import set_seed


def main():
    """Main function to run experiments"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DQN on Ms. Pac-Man')
    
    # Experiment settings
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['eval', 'full'],
                       help='Mode: eval only, or full experiment')
    parser.add_argument('--env', type=str, default='MsPacman',
                       help='Environment name')
    parser.add_argument('--num_seeds', type=int, nargs='+', default=1,
                       help='Number of seeds for training')
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Total training steps per seed')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    
    # Output settings
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*60)
    print("MS. PAC-MAN DQN EXPERIMENT")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Environment: {args.env}")
    print(f"Nunber of seeds: {args.num_seeds}")
    print(f"Training steps: {args.steps:,}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Save directory: {args.save_dir}")
    print(f"Device: {args.device}")
    print("="*60)
    print()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)    
       
    if args.mode == 'eval':
        # Evaluate pre-trained agent
        print("\nEvaluation mode - load trained model and evaluate")
        # TODO: Implement loading and evaluating saved model
        print("Not implemented yet. Use --mode full for complete workflow.")
        
    elif args.mode == 'full':
        # Run full experiment: train multiple seeds + evaluate each
        print("\nRunning full experiment...")
        run_full_experiment(
            env_name=args.env,
            num_seeds=len(args.num_seeds),
            training_steps=args.steps,
            eval_episodes=args.eval_episodes,
            save_dir=args.save_dir,
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


#!/bin/bash

pip install -r requirements.txt
python main.py --mode full --save-dir my_experiment_results

"""

