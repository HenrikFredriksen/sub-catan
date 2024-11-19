import numpy as np

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.CatanEnv import CatanEnv

'''
This script tests the CatanEnv environment by taking random actions for each agent
until all agents are done or a maximum number of steps is reached.
Use as a reference to understand how to interact with the environment.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
def test_environment():
    env = CatanEnv(render_mode='human', gamestate='normal_phase')
    env.reset()
    
    max_steps = 5000
    step_count = 0
    
    while True:
        agent = env.agent_selection
        obs, rew, termination, truncation, info = env.last()
        done = termination or truncation
        print(f"Step: {step_count} ------------------------------------------------")
        #print(f"gamestate: {env.game_manager.gamestate}")
        #print(f"Agent: {agent}")
        #print(f"Observation: {obs}")
        #print(f"Reward: {rew}, Done: {done}")
        
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
        