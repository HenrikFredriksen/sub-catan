import os
import time
import traceback
import torch
import imageio

from .MultiAgentPPOImpl import MultiAgentPPO
from environment.CatanEnv import CatanEnv
from config import get_default_config
'''
Contains the evaluation script for evaluating tested agents in the Catan environment.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
class Evaluator:
    def __init__(self, config, args=None):
        self.cfg = config
        self.env = None
        self.writer = None
        self.verbose = self.cfg.miscs.verbose
        self.model_path = self.cfg.test.model_path

        self.batch_size = self.cfg.test.batch_size
        self.num_episodes = self.cfg.test.num_episodes
        self.learning_rate = self.cfg.test.learning_rate
        self.gamma = self.cfg.test.gamma
        self.gae_lambda = self.cfg.test.gae_lambda
        self.clip_epsilon = self.cfg.test.clip_epsilon
        self.n_epochs = self.cfg.test.n_epochs
        self.max_steps = self.cfg.test.max_steps
        self.hidden_dim = self.cfg.test.hidden_dim
        self.agent_policies = self.cfg.test.agent_policies

    def eval_trained_agents(self, render_mode='human', gamestate='normal_phase'):
        os.makedirs("game_frames", exist_ok=True)
        self.env = CatanEnv(
            render_mode=render_mode,
            gamestate=gamestate,
            verbose=self.verbose,

        )

        frames = []
        self.env.reset()

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

        for agent_id in ppo.agents:
            evaluation_model_path = f"{self.model_path}run_015_e10k/best_model_agent_{agent_id}.pt"
            if os.path.exists(evaluation_model_path):
                ppo.agents[agent_id]["network"].load_state_dict(
                    torch.load(evaluation_model_path)
                )
                print(f"Loaded model for agent {agent_id}")

        done = False
        frame_count = 0

        while not done:
            if all(self.env.terminations.values()):
                done = True
                self.env.close()
                print("All agents have been terminated")
                break
            
            try:
                observation, reward, termination, truncation, info = self.env.last()
                if termination or truncation:
                    action = None
                else:
                    action = ppo.choose_action(self.env.agent_selection, observation)[0]
                self.env.step(action)

                if render_mode == 'rgb_array':
                    frame = self.env.render()
                    if frame is not None:
                        frames.append(frame)

                    frame_count += 1

            except Exception as e:
                print(f"Error in evaluation loop: {e}")
                print(f"agent_selection: {self.env.agent_selection}")
                traceback.print_exc()
                break
            
        self.env.close()

        if frames and render_mode == 'rgb_array':
            print("Saving frames to gif")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            imageio.mimsave(
                f"game_frames/game_{timestamp}.gif", 
                frames,
                fps=2
            )
            print(f"Game saved as game/frames/catangame_{timestamp}.gif")

if __name__ == "__main__":
    #pretest_settlement_phase()
    #main()
    evaluator = Evaluator(config=get_default_config())
    for _ in range(evaluator.cfg.test.num_evals):
        evaluator.eval_tested_agents()

