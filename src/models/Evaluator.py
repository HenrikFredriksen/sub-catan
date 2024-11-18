import os
import time
import traceback
import torch
import imageio

from .MultiAgentPPOImpl import MultiAgentPPO
from environment.CatanEnv_torch_spec import CatanEnv

def eval_trained_agents(render_mode='human', gamestate='normal_phase'):
    os.makedirs("game_frames", exist_ok=True)
    env = CatanEnv(render_mode=render_mode, gamestate=gamestate)
    
    frames = []
    env.reset()
    
    ppo = MultiAgentPPO(
        env=env,
        writer=None,
        hidden_dim=1536,
        batch_size=32,
        learning_rate=0.0002,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.4,
        n_epochs=4,
        max_steps=10000
    )
    
    for agent_id in ppo.agents:
        pretrained_path = f"best_model_agent_{agent_id}.pt"
        if os.path.exists(pretrained_path):
            ppo.agents[agent_id]["network"].load_state_dict(
                torch.load(pretrained_path)
            )
            print(f"Loaded model for agent {agent_id}")
    
    done = False
    frame_count = 0
    
    while not done:
        if all(env.terminations.values()):
            done = True
            env.close()
            print("All agents have been terminated")
            break
        
        try:
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                action = ppo.choose_action(env.agent_selection, observation)[0]
            env.step(action)
            
            if render_mode == 'rgb_array':
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

                frame_count += 1
                
        except Exception as e:
            print(f"Error in evaluation loop: {e}")
            print(f"agent_selection: {env.agent_selection}")
            traceback.print_exc()
            break
        
    env.close()
        
    if frames and render_mode == 'rgb_array':
        print("Saving frames to gif")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        imageio.mimsave(
            f"game_frames/game_{timestamp}.gif", 
            frames,
            fps=12
        )
        print(f"Game saved as game/frames/catangame_{timestamp}.gif")

if __name__ == "__main__":
    #pretrain_settlement_phase()
    #main()
    
    num_evals = 10
    for _ in range(num_evals):
        eval_trained_agents()
    
