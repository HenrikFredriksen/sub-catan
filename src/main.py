import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Evaluator import eval_trained_agents
from models.Trainer import train, train_settlement_phase
from gameloop.GameLoop import GameLoop
from models.test_catan_env_random_action import test_environment

def argparser():
    parser = argparse.ArgumentParser(description="Train Catan agents")
    parser.add_argument(
        "--train_settle_phase",
        type=bool,
        default=False,
        help="Train agents in the settlement phase of the game",
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="Train agents in the normal phase of the game",
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=0,
        help="Evaluate trained agents, how many times to evaluate",
    )
    parser.add_argument(
        "--test_env",
        type=bool,
        default=False,
        help="Test the environment by taking random actions for each agent",
    )
    parser.add_argument(
        "--interactive",
        type=bool,
        default=False,
        help="Run the interactive game loop",
    )
    return parser.parse_args()

def main():
    args = argparser()
    
    if args.train_settle_phase:
        train_settlement_phase()
    elif args.train:
        train(n_episodes=2500)
    elif args.eval != 0:
        for _ in range (args.eval):
            eval_trained_agents(render_mode='human', gamestate='normal_phase')
    elif args.test_env:
        test_environment()
    elif args.interactive:
        GameLoop().main()
    else:
        print("No valid argument provided. Use --help for more information.")


if __name__ == "__main__":
    '''
    this is the main entry point for the project.
    uncomment the function you want to run.
    '''
    main()