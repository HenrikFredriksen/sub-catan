from environment.CatanEnv import CatanEnv
import numpy as np

def test_environment():
    env = CatanEnv()
    env.reset()
    
    max_steps = 100
    step_count = 0
    
    while True:
        agent = env.agent_selection
        obs, rew, termination, truncation, info = env.last()
        done = termination or truncation
        print(f"Agent: {agent}")
        print(f"Observation: {obs}")
        print(f"Reward: {rew}, Done: {done}")
        
        if done:
            action = None
            print("Agent done, skipping action")
        else:
            valid_actions = env.get_valid_actions(agent)
            print(f"Valid actions: {valid_actions}")
            if valid_actions:
                action = np.random.choice(valid_actions)
                print(f"Taking action: {action}")
            else:
                action = None
                print("No valid actions, skipping action")
                
        env.step(action)
        step_count += 1
        
        all_terminated = all(env.terminations.values())
        all_truncated = all(env.truncations.values())
        if all_terminated or all_truncated or step_count >= max_steps:
            print("All agents done or max steps reached")
            break
        
    env.close()
    
if __name__ == "__main__":
    test_environment()
        