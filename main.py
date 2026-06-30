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
    parser.add_argument('--seeds', type=int, default=42,
                       help='Seed for runing exp')
    parser.add_argument('--steps', type=int, default=1_000_000,
                       help='Total training steps per seed')
    
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
       
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {args.save_dir}")
    print("="*60)

if __name__ == "__main__":
    main()


# In[ ]:




