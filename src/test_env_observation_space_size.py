from environment.CatanEnv import CatanEnv

if __name__ == "__main__":
    env = CatanEnv()
    env.reset()
    for agent in env.agent_iter():
        observation = env.observe(agent)
        print(f"observation for {agent}: {observation}")
        action = env.action_space(agent).sample()
        env.step(action)