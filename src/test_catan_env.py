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
        print(f"Step: {step_count} ------------------------------------------------")
        print(f"Agent: {agent}")
        print("Observation: \n " +
              f"Player: {obs['player_state']} \n" +
              f"Enemy: {obs['enemy_state']}")
        print(f"Reward: {rew}, Done: {done}")
        
        if done:
            action = env.pass_action_index
            print("Agent done, passing turn")
        else:
            valid_actions = env.get_valid_actions(agent)
            print(f"Valid actions: {valid_actions}")
            action = np.random.choice(valid_actions)
            print(f"Taking action: {action}")
                
        env.step(action)
        step_count += 1
        

        
        if all(env.terminations.values()) or step_count >= max_steps:
            print("All agents done or max steps reached")
            break
        
    env.close()
    
if __name__ == "__main__":
    test_environment()
        