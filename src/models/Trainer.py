import os
import torch

import argparse

from torch.utils.tensorboard import SummaryWriter
from src.models.MultiAgentPPOImpl import MultiAgentPPO
from environment.CatanEnv import CatanEnv
from environment.CatanSettlePhaseEnv import CatanSettlePhaseEnv 
from assets.NullWriter import NullWriter

'''
This file contains the training scripts for training agents in different phases of the game.

train_settlement_phase() - Train agents in the settlement phase of the game
train() - Train agents in either the normal phase or the whole game

Both scripts support multiple policies for different agents.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class Trainer:
    def __init__(self, config, perf: bool = False):
        self.cfg = config
        self.perf = perf
        self.writer = NullWriter() if perf else SummaryWriter(log_dir=self.cfg.miscs.log_dir)
        self.verbose = False if perf else self.cfg.miscs.verbose
        
        self.env = None

        self.seed = self.cfg.miscs.seed
        self.batch_size = self.cfg.train.batch_size
        self.num_episodes = self.cfg.train.num_episodes
        self.learning_rate = self.cfg.train.learning_rate
        #self.eval_interval = self.cfg.eval.eval_interval
        self.gamma = self.cfg.train.gamma
        self.gae_lambda = self.cfg.train.gae_lambda
        self.clip_epsilon = self.cfg.train.clip_epsilon
        self.n_epochs = self.cfg.train.n_epochs
        self.max_steps = self.cfg.train.max_steps
        self.hidden_dim = self.cfg.train.hidden_dim
        self.agent_policies = self.cfg.train.agent_policies

    def train_settlement_phase(self, n_episodes=5000):
        self.env = CatanSettlePhaseEnv(writer=self.writer)

        ppo = MultiAgentPPO(
            env=self.env,
            writer=self.writer,
            agent_policies=self.agent_policies,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            n_epochs=self.n_epochs,
            max_steps=self.max_steps
        )

        rewards = ppo.train(n_episodes, seed=self.seed)

        for agent_id in ppo.agents:
            torch.save(
                ppo.agents[agent_id]["network"].state_dict(), 
                f"pretrained_settle_{agent_id}.pt"
            )

        self.writer.close()
        return ppo

    def train(self, gamestate='normal_phase', n_episodes=1, seed=1234, max_env_calls=None):
        # Change gamestate 'settle_phase' to run whole game training
        # Change gamestate 'normal_phase' to run normal phase training with loaded board
        self.env = CatanEnv(gamestate=gamestate, render_mode=self.cfg.miscs.render_mode, verbose=self.verbose)

        # Init PPO agent
        ppo = MultiAgentPPO(
            env=self.env,
            writer=self.writer,
            agent_policies=self.agent_policies,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_epsilon=self.clip_epsilon,
            n_epochs=self.n_epochs,
            max_steps=self.max_steps
        )

        if not self.perf:
            for agent_id, agent_data in ppo.agents.items():
                pretrained_path = f"{self.cfg.train.pretrained_model_path}best_model_agent_{agent_id}.pt"
                if os.path.exists(pretrained_path):
                    try:
                        ppo.agents[agent_id]["network"].load_state_dict(
                            torch.load(pretrained_path, weights_only=True)
                        )
                        print(f"Loaded pretrained model for agent {agent_id}")
                    except Exception as e:
                        print(f"Error loading pretrained model for agent {agent_id}: {e}")

                model = agent_data['network']
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                param_message = f"{agent_id} parameter count: {total_params}"

                self.writer.add_text("Model/ParameterCount", param_message)

                print(param_message)

        # Train the agent
        ppo.train(n_episodes, 
                  seed=seed, 
                  max_env_calls=max_env_calls,
                  verbose=self.verbose,
                  perf=self.perf)
        
        self.writer.close()
        # print("Training completed!")
        return ppo


