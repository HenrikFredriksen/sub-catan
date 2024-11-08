import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from env.CatanEnv import CatanEnv

def env_creator(env_config):
    return CatanEnv()

ray.init()

tune.register_env("catan", env_creator)

config = {
    "env": "catan",
    "framework": "tf",
    "num_gpus": 1,
    "num_workers": 0,
    "multiagent": {
        "policies": {
            "player_1": (None, CatanEnv.observation_spaces['player1'], CatanEnv.action_spaces['player1'], {}),
            "player_2": (None, CatanEnv.observation_spaces['player2'], CatanEnv.action_spaces['player2'], {}),
            "player_3": (None, CatanEnv.observation_spaces['player3'], CatanEnv.action_spaces['player3'], {}),
            "player_4": (None, CatanEnv.observation_spaces['player4'], CatanEnv.action_spaces['player4'], {}),
        },
        "policy_mapping_fn": lambda agent_id: "agent_id",
    },
}

trainer = PPOTrainer(env="catan", config=config)

for i in range(1000):
    result = trainer.train()
    print(f"iteration: {i}, reward: {result['episode_reward_mean']}")
    
    if i % 50 == 0:
        checkpoint = trainer.save()
        print(f"checkpoint saved at {checkpoint}")
        
ray.shutdown()