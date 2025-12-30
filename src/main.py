import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Evaluator import Evaluator
from models import Trainer
from gameloop.GameLoop import GameLoop
from models.test_catan_env_random_action import test_environment
from config import get_default_config, parse_config_file, Config

def argparser():
    parser = argparse.ArgumentParser(description="Train Catan agents")
    parser.add_argument(
        "-f",
        "--config_file",
        default=None,
        type=str,
        help="Path to the config file for training",
    )
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
        type=bool,
        default=False,
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
    parser.add_argument(
        "--cfg_opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line",
    )
    return parser.parse_args()

def main():
    args = argparser()
    cfg = parse_config_file(args.config_file) if args.config_file else get_default_config() 
    if args.config_file:
        # Parse the config file
        cfg.merge(args.cfg_opts)
    
    trainer = Trainer.Trainer(
        config=cfg,
    )

    match True:
        case _ if args.train_settle_phase:
            print("Training settle phase with config:", cfg)
            trainer.train_settlement_phase()
        case _ if args.train:
            print("Training with config: \n", cfg)
            trainer.train(n_episodes=cfg.train.num_episodes, 
                          seed=cfg.miscs.seed, 
                          gamestate=cfg.train.gamestate)
        case _ if args.eval:
            evaluator = Evaluator(cfg)
            for _ in range (cfg.test.num_evals):
                evaluator.eval_trained_agents(render_mode=cfg.eval.render_mode, 
                                              gamestate='normal_phase')
        case _ if args.test_env:
            test_environment()
        case _ if args.interactive:
            GameLoop().main()
        case _:
            print("No valid argument provided. Use --help for more information.")

if __name__ == "__main__":
    '''
    this is the main entry point for the project.
    uncomment the function you want to run.
    '''
    main()