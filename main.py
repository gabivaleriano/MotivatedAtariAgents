#!/usr/bin/env python
# coding: utf-8

# In[1]:


from training import train_with_seed
from utils import set_seed

import argparse

def main():
    """Main function to run experiments"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DQN on Ms. Pac-Man')
    
    # Experiment settings
    parser.add_argument('--seed', type=int, default=42,
                       help='Seed for runing exp')
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Total training steps per seed')
    parser.add_argument('--incentive', type=bool, default=False,
                       help='Call the function training_with_seed_incentive')
    
    # Output settings
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()

    # Print configuration
    print("="*60)
    print("MS. PAC-MAN DQN EXPERIMENT")
    print("="*60)
    print(f"Seeds: {args.seed}")
    print(f"Training steps: {args.steps:,}")
    print(f"Save directory: {args.save_dir}")
    print("="*60)
    print()

    train_with_seed(
    seed=args.seed,
    steps=args.steps,
    save_dir=args.save_dir,
    incentive=args.incentive)
       
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {args.save_dir}")
    print("="*60)

if __name__ == "__main__":
    main()


# In[ ]:




