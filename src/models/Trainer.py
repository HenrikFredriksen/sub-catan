import os
import torch

from torch.utils.tensorboard import SummaryWriter
from src.models.MultiAgentPPOImpl import MultiAgentPPO
from environment.CatanEnv import CatanEnv
from environment.CatanSettlePhaseEnv import CatanSettlePhaseEnv 

def pretrain_settlement_phase():
    writer = SummaryWriter(log_dir='run_logs/catan_settle_pretraining')
    env = CatanSettlePhaseEnv(writer=writer)
    writer = SummaryWriter(log_dir='run_logs/catan_settle_pretraining')
    
    agent_policies = {
        'player_1': 'baseline',
        'player_2': 'baseline',
        'player_3': 'baseline',
        'player_4': 'baseline'
    }

    ppo = MultiAgentPPO(
        env=env,
        writer=writer,
        agent_policies=agent_policies,
        hidden_dim=1536,
        batch_size=32,
        learning_rate=0.0002,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.4,
        n_epochs=4,
        max_steps=5000
    )

    n_episodes = 500
    base_seed = 42
    rewards = ppo.train(n_episodes, seed=base_seed)

    for agent_id in ppo.agents:
        torch.save(
            ppo.agents[agent_id]["network"].state_dict(), 
            f"pretrained_settle_{agent_id}.pt"
        )

    writer.close()
    return ppo

def train(gamestate='normal_phase', n_episodes=2500):
    # Change gamestate 'settle_phase' to run whole game training
    # Change gamestate 'normal_phase' to run normal phase training with loaded board
    env = CatanEnv(gamestate=gamestate)
    
    writer = SummaryWriter(log_dir='run_logs/catan_training')
    
    agent_policies = {
        'player_1': 'baseline',
        'player_2': 'baseline',
        'player_3': 'baseline',
        'player_4': 'baseline'
    }
    
    # Init PPO agent
    ppo = MultiAgentPPO(
        env=env,
        writer=writer,
        agent_policies=agent_policies,
        hidden_dim=1536,
        batch_size=32,
        learning_rate=0.0002,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.4,
        n_epochs=4,
        max_steps=5000
    )

    for agent_id in ppo.agents:
        pretrained_path = f"best_model_agent_player_{agent_id}.pt"
        if os.path.exists(pretrained_path):
            ppo.agents[agent_id]["network"].load_state_dict(
                torch.load(pretrained_path)
            )
            print(f"Loaded pretrained model for agent {agent_id}")

    # Train the agent
    base_seed = 42
    rewards = ppo.train(n_episodes, seed=base_seed, max_turns_without_building=2000)
    
    writer.close()
    
    # Plot training rewards if desired
    print("Training completed!")
    print(f"Final average reward: {sum(rewards) / len(rewards):.2f}")
    return ppo
