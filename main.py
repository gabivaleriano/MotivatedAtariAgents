#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Main entry point for Ms. Pac-Man DQN experiments
Run training with multiple seeds and evaluation
"""
import argparse
import sys
import os
from experiment import train_with_seed, run_full_experiment, evaluate_agent
from utils import set_seed
from loguru import logger

def setup_logging(save_dir: str):
    """Configure loguru to log to both console and file."""
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "experiment.log")

    # Remove default stderr sink
    logger.remove()

    # Console: INFO and above
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")

    # File: DEBUG and above, with full tracebacks on crashes
    logger.add(
        log_path,
        level="DEBUG",
        rotation="50 MB",        # start a new file after 50 MB
        retention=3,             # keep last 3 rotated files
        backtrace=True,          # full traceback on exceptions
        diagnose=True,           # show variable values in traceback
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )

    logger.info(f"Logging to: {log_path}")


def main():
    """Main function to run experiments"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DQN on Ms. Pac-Man')
    
    # Experiment settings
    parser.add_argument('--env', type=str, default='MsPacman',
                       help='Environment name')
    parser.add_argument('--num_seeds', type=int, default=5,
                       help='Number of seeds for training')
    parser.add_argument('--steps', type=int, default=2_000_000,
                       help='Total training steps per seed')
    parser.add_argument('--eval-episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--clip_rewards', action='store_true', default=False,
                   help='Clip rewards to -1, 0, 1')
    parser.add_argument('--agent_styles', type=str, nargs ='+',  default= ['Incentive'],
                       choices = ['Vanilla', 'Hull', 'WantLike', 'Incentive'],
                       help='List with agents to be trained. Options: Vanilla, Hull, Want_like, Incentive')
    
    # Output settings
    parser.add_argument('--save-dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()

    # Set up logging FIRST, before anything else runs
    setup_logging(args.save_dir)

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

    logger.info("=" * 60)
    logger.info("MS. PAC-MAN DQN EXPERIMENT")
    logger.info(f"Environment:         {args.env}")
    logger.info(f"Number of seeds:     {args.num_seeds}")
    logger.info(f"Training steps:      {args.steps:,}")
    logger.info(f"Evaluation episodes: {args.eval_episodes}")
    logger.info(f"Agent styles:        {args.agent_styles}")
    logger.info(f"Save directory:      {args.save_dir}")
    logger.info("=" * 60)

    try:
        logger.info("Running full experiment...")
        run_full_experiment(
            env_name=args.env,
            num_seeds=args.num_seeds,
            training_steps=args.steps,
            eval_episodes=args.eval_episodes,
            save_dir=args.save_dir,
            agent_styles=args.agent_styles,
            clip_rewards=args.clip_rewards,
        )
        logger.info("EXPERIMENT COMPLETE! Results saved to: {}", args.save_dir)

    except Exception:
        # logger.exception captures the full traceback automatically
        logger.exception("Experiment crashed with an unhandled exception!")
        sys.exit(1)  # non-zero exit so the cluster scheduler marks the job as failed  
        
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


# In[ ]:




