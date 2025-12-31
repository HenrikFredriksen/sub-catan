'''
NOTE: The structure and logic for this config class are partly adapted from
https://github.com/tinyvision/DAMO-YOLO/blob/master/damo/config/base.py
(Apache License 2.0). Many thanks to the original authors for open-sourcing
their configuration system.
'''

import ast
import importlib
import os
import pprint
import sys
from abc import ABCMeta
from os.path import dirname, join

from easydict import EasyDict as easydict
from tabulate import tabulate

agent_policies = {
    'player_1': 'alternative',
    'player_2': 'baseline',
    'player_3': 'alternative',
    'player_4': 'baseline'
}


miscs = easydict({
    'print_interval_iters': 50,
    'seed': 42,
    'experiment_name': 'catan',
    'log_dir': './run_logs/catan_training/',
    'render_mode': 'human',
    'verbose': True,
})

train = easydict({
    'batch_size': 128,
    'num_episodes': 25000,
    'learning_rate': 0.0002,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.1,
    'n_epochs': 4,
    'max_steps': 10000,
    'hidden_dim': 256,
    'agent_policies': agent_policies,
    'pretrained_model_path': './saved_models/run_015/',
    'gamestate': 'settle_phase',
})

eval = easydict({
    'eval_interval': 100,
    'render_mode': 'human',
    'num_evals': 3,
    'model_path': './saved_models/run_007/best_models/',
})

test = easydict({
    'batch_size': 1,
    'num_episodes': 1,
    'learning_rate': 0.0002,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.4,
    'n_epochs': 4,
    'max_steps': 10000,
    'hidden_dim': 1536,
    'agent_policies': agent_policies,
    'render_mode': 'human',
    'num_evals': 1,
    'model_path': './saved_models/',
})

playingboards = easydict({
    'boards_file_path': 'normal_phase_boards',
})


class Config(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.miscs = miscs
        self.train = train
        self.test = test
        self.eval = eval
        self.playingboards = playingboards
        self.model = easydict()

    def __repr__(self):
        table_header = ['keys', 'values']
        exp_table = [(str(k), pprint.pformat(v, compact=True))
                     for k, v in vars(self).items() if not k.startswith('_')]
        return tabulate(exp_table, headers=table_header, tablefmt='fancy_grid')

    def merge(self, cfg_list):
        """
        Merge config from a list of [key, value, key, value, ...] pairs.
        Example usage:
            cfg.merge(['train.batch_size', 64, 'miscs.seed', 999])
        """
        if len(cfg_list) % 2 != 0:
            raise ValueError("cfg_list length must be even (key-value pairs).")
        for key, value in zip(cfg_list[0::2], cfg_list[1::2]):
            # e.g. "train.batch_size" => sub_keys=["train"], final_key="batch_size"
            *sub_keys, final_key = key.split('.')
            cur_obj = self
            for sk in sub_keys:
                if not hasattr(cur_obj, sk):
                    raise KeyError(f"Config has no attribute '{sk}'")
                cur_obj = getattr(cur_obj, sk)

            if not hasattr(cur_obj, final_key):
                raise KeyError(f"Config has no attribute '{final_key}' in '{'.'.join(sub_keys)}'")

            original_val = getattr(cur_obj, final_key)
            original_type = type(original_val)

            # Attempt type casting or literal_eval if the types differ
            if (original_val is not None) and (original_type != type(value)):
                try:
                    if original_type is bool:
                        # accept 1/0, true/false, yes/no (case‑insensitive)
                        from distutils.util import strtobool
                        value = bool(strtobool(str(value)))
                    else:
                        value = original_type(value)
                except (ValueError, SyntaxError):
                    value = ast.literal_eval(str(value))

            setattr(cur_obj, final_key, value)


def parse_config_file(config_file):

    if not config_file:
        raise ValueError("config file path cannot be empty or None")
    sys.path.append(os.path.dirname(config_file))
    cfg_module_name = os.path.basename(config_file).split('.')[0]
    try:
        current_config = importlib.import_module(cfg_module_name)
        exp = current_config.Config()
    except Exception as e:
        raise ImportError(
            "{} doesn't contains class named 'Config': {}".format(config_file, str(e)))
    return exp


def get_default_config():
    return Config()