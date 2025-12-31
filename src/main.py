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
        action="store_true",
        help="Train agents in the settlement phase of the game",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train agents in the normal phase of the game",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate trained agents, how many times to evaluate",
    )
    parser.add_argument(
        "--test_env",
        action="store_true",
        help="Test the environment by taking random actions for each agent",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run the interactive game loop",
    )

    # performance optimization args
    parser.add_argument(
        "--perf", 
        action="store_true",
        help="Run performance benchmark mode (prints PERF_SUMMARY)")
    parser.add_argument(
        "--max_env_calls", 
        type=int, 
        default=None,
        help="Stop after this many env.step calls (benchmarking)")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Override seed (optional)")
    
    # config override args
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
    if args.cfg_opts:
        # Parse the config file
        cfg.merge(args.cfg_opts)

    if args.perf:
        cfg.miscs.verbose = False
        cfg.miscs.render_mode = None
    
    trainer = Trainer.Trainer(config=cfg, perf=args.perf)

    match True:
        case _ if args.train_settle_phase:
            print("Training settle phase with config:", cfg)
            trainer.train_settlement_phase()
        case _ if args.train:
            if not args.perf:
                print("Training with config: \n", cfg)

            seed = args.seed if args.seed is not None else cfg.miscs.seed

            trainer.train(n_episodes=cfg.train.num_episodes, 
                          seed=seed, 
                          gamestate=cfg.train.gamestate,
                          max_env_calls=args.max_env_calls if args.perf else None)
            
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