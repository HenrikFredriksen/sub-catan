import os
import torch

import argparse

from torch.utils.tensorboard import SummaryWriter
from src.models.MultiAgentPPOImpl import MultiAgentPPO
from environment.CatanEnv import CatanEnv
from environment.CatanSettlePhaseEnv import CatanSettlePhaseEnv 

'''
This file contains the training scripts for training agents in different phases of the game.

train_settlement_phase() - Train agents in the settlement phase of the game
train() - Train agents in either the normal phase or the whole game

Both scripts support multiple policies for different agents.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class Trainer:
    def __init__(self, config, args=None):
        self.cfg = config
        #self.args = args
        self.env = None
        self.writer = SummaryWriter(log_dir=self.cfg.miscs.log_dir)
        self.verbose = config.miscs.verbose 

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

        
    # Argument parser
    def make_parser():
        parser = argparse.ArgumentParser(description="Train Catan agents")
        parser.add_argument(
            "--gamestate",
            type=str,
            default="normal_phase",
            help="Gamestate to train on: 'normal_phase' or 'settle_phase'",
        )
        parser.add_argument(
            "--n_episodes", type=int, default=5000, help="Number of episodes to train"
        )
        parser.add_argument(
            "--render_mode",
            type=str,
            default="human",
            help="Render mode for evaluation: 'human' or 'rgb_array'",
        )
        parser.add_argument(
            "--num_evals",
            type=int,
            default=3,
            help="Number of evaluations to run after training",
        )

        return parser.parse_args()

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

        n_episodes = 500
        rewards = ppo.train(n_episodes, seed=self.seed, max_turns_without_building=1000)

        for agent_id in ppo.agents:
            torch.save(
                ppo.agents[agent_id]["network"].state_dict(), 
                f"pretrained_settle_{agent_id}.pt"
            )

        self.writer.close()
        return ppo

    def train(self, gamestate='normal_phase', n_episodes=1, seed=1234):
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



        for agent_id, agent_data in ppo.agents.items():
            #pretrained_path = f"{self.cfg.train.pretrained_model_path}best_model_agent_player_{agent_id}.pt"
            #if os.path.exists(pretrained_path):
            #    ppo.agents[agent_id]["network"].load_state_dict(
            #        torch.load(pretrained_path)
            #    )
            #    print(f"Loaded pretrained model for agent {agent_id}")

            model = agent_data['network']
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            message = f"{agent_id} parameter count: {total_params}"

            self.writer.add_text("Model/ParameterCount", message)

            print(message)

        # Train the agent
        rewards = ppo.train(n_episodes, seed=seed, max_turns_without_building=1000)

        self.writer.close()

        # Plot training rewards if desired
        print("Training completed!")
        print(f"Final average reward: {sum(rewards) / len(rewards):.2f}")
        return ppo


