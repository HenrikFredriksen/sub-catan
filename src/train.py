import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF2Policy
from ray.rllib.utils.framework import try_import_tf
from environment.CatanEnv import CatanEnv
from ray.rllib.env import PettingZooEnv

tf1, tf, tfv = try_import_tf()

class CustomPPOTF2Policy(PPOTF2Policy):
    def optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=float(self.config["lr"]))

def env_creator(env_config):
    env = CatanEnv()
    return PettingZooEnv(env)

ray.init()

tune.register_env("catan", env_creator)

env_instance = CatanEnv()
env = PettingZooEnv(env_instance)

ppo_config = (
    PPOConfig()
    .environment("catan")
    .framework("tf2")
    .env_runners(num_env_runners=0)
    .multi_agent(
        policies={
            "shared_policy": (
                CustomPPOTF2Policy, 
                env.observation_space, 
                env.action_space, 
                {}
            ),
        },
        policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
    )
    .training(
        lr=0.0001,
    )
    .resources(
        num_gpus=1,
    )
)

algo = ppo_config.build()



for i in range(100):
    result = algo.train()
    print(f"iteration: {i}, reward: {result['episode_reward_mean']}")
    
    if i % 10 == 0:
        checkpoint = algo.save()
        print(f"checkpoint saved at {checkpoint}")
        
ray.shutdown()