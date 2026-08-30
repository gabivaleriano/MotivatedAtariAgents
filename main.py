#!/usr/bin/env python
# coding: utf-8

# In[1]:


from training import complete_training
from utils import set_seed

import argparse

def main():
    """Main function to run experiments"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DQN on Ms. Pac-Man')
    
    # Experiment settings
    parser.add_argument('--num_seeds', type=int, default=1,
                       help='Number of seeds for training')
    parser.add_argument('--steps', type=int, default=3_500_000,
                       help='Total training steps per seed')
    parser.add_argument('--agents', type=str, nargs ='+',  default= ['Incentive'],
                       choices = ['Vanilla', 'Incentive','Hull','WantLike'],
                       help='List with agents to be trained. Options: Vanilla, Incentive')
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number episodes for evaluation')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Modulation intensity')
    parser.add_argument('--loss', type=int, default=100,
                       help='Life loss penalty')
    parser.add_argument('--kappa_', type=int, default=0,
                       help='Kappa modulation, can receive 1 to set off')
    parser.add_argument('--like', type=int, default=1,
                       help='Like reward on Incentive agent')
    parser.add_argument('--dqn_modulation', type=int, default=1,
                       help='DQN modulation, can receive 0 to hand-coded modulation')

    # Output settings
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()

    # Print configuration
    print("="*60)
    print("MS. PAC-MAN DQN EXPERIMENT")
    print("="*60)
    print(f"Number of seeds: {args.num_seeds}")
    print(f"Training steps: {args.steps:,}")
    print(f"Save directory: {args.save_dir}")
    print("="*60)
    print()

    complete_training(
    num_seeds=args.num_seeds,
    steps=args.steps,
    alpha = args.alpha,
    save_dir=args.save_dir,
    agents=args.agents,
    loss=args.loss,
    kappa_=args.kappa_,
    like=args.like,
    dqn_modulation=args.dqn_modulation,
    eval_episodes=args.eval_episodes)
       
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print(f"Results saved to: {args.save_dir}")
    print("="*60)

if __name__ == "__main__":
    main()


# In[ ]:




