import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from collections import deque
import time

from environment.CatanEnv_torch_spec import CatanEnv

class PPONetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PPONetwork, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, action_mask):
        shared_features = self.shared_layers(x)
        
        # Apply action mask to policy logits
        policy = self.policy_head(shared_features)
        masked_policy = policy * action_mask
        masked_sum = masked_policy.sum(dim=-1, keepdim=True)

        masked_sum = torch.where(masked_sum == 0, torch.ones_like(masked_sum), masked_sum)

        policy = masked_policy / masked_sum
        
        value = self.value_head(shared_features)
        
        return policy, value

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.batch_size = batch_size
        
    def store(self, state, action, prob, val, reward, done, action_mask):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.vals.clear()
        self.rewards.clear()
        self.dones.clear()
        self.action_masks.clear()
        
    def get_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return (
            self.states,
            self.actions,
            self.probs,
            self.vals,
            self.rewards,
            self.dones,
            self.action_masks,
            batches
        )

class MultiAgentPPO:
    def __init__(
        self,
        env,
        hidden_dim=256,
        batch_size=32,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        c1=1.0,  # Value function coefficient
        c2=0.01,  # Entropy coefficient
        n_epochs=4,
        max_steps=10000 
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.n_epochs = n_epochs
        self.max_steps = max_steps
        
        # Init networks and optimizers for each agent
        self.agents = {}
        self.memories = {}
        
        for agent_id in self.env.possible_agents:
            obs_dim = env.observation_spaces[agent_id]["observation"].shape[0]
            act_dim = env.action_spaces[agent_id].n
            
            # Create network and optimizer
            network = PPONetwork(obs_dim, hidden_dim, act_dim)
            optimizer = optim.Adam(network.parameters(), lr=learning_rate)
            
            self.agents[agent_id] = {
                "network": network,
                "optimizer": optimizer
            }
            # Create memory
            self.memories[agent_id] = PPOMemory(batch_size)
    
    def choose_action(self, agent_id, observation):
        state = torch.FloatTensor(self.env.observe(agent_id)["observation"])
        action_mask = torch.FloatTensor(self.env.observe(agent_id)["action_mask"])
        
        with torch.no_grad():
            if torch.sum(action_mask) == 0:
                print(f"Warning: No valid actions available agent {agent_id}")
                return 0, float('-inf'), 0.0
            policy, value = self.agents[agent_id]["network"](state, action_mask)


            if torch.isnan(policy).any():
                print(f"Warning: NaN in policy for agent {agent_id}")
                # Return a dummy action
                return 0, float('-inf'), value.item()
            
            # Create distribution only for valid actions
            dist = Categorical(policy)
            action = dist.sample()
            prob = dist.log_prob(action)
        
        return action.item(), prob.item(), value.item()
    
    def learn(self, agent_id):
        memory = self.memories[agent_id]
        network = self.agents[agent_id]["network"]
        optimizer = self.agents[agent_id]["optimizer"]
        
        (
            states,
            actions,
            old_probs,
            vals,
            rewards,
            dones,
            action_masks,
            batches
        ) = memory.get_batches()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        old_probs = torch.FloatTensor(np.array(old_probs))
        vals = torch.FloatTensor(np.array(vals))
        rewards = torch.FloatTensor(np.array(rewards))
        dones = torch.FloatTensor(np.array(dones))
        action_masks = torch.FloatTensor(np.array(action_masks))
        
        # Calculate advantages
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0
            else:
                next_val = vals[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - vals[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lam
            
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            for batch in batches:
                # Get batch data
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_advantages = advantages[batch]
                batch_action_masks = action_masks[batch]
                
                # Forward pass
                policy, value = network(batch_states, batch_action_masks)
                
                policy = policy + 1e-10
                policy = policy / policy.sum(dim=-1, keepdim=True)

                try:
                    # Calculate policy loss
                    dist = Categorical(policy)
                    new_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()
                except:
                    print(f"Invalid policy distribution: {policy.sum(dim=-1)}")
                    continue
                
                ratio = torch.exp(new_probs - batch_old_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_target = batch_advantages + vals[batch]
                value_loss = ((value.squeeze() - value_target) ** 2).mean()
                
                # Total loss
                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        memory.clear()

    def train(self, n_episodes):
        best_reward = float('-inf')
        episode_rewards = deque(maxlen=100)
        
        for episode in range(n_episodes):
            print(f"Starting episode {episode + 1}")
            episode_reward = {agent: 0 for agent in self.env.possible_agents}
            self.env.reset()
            agent_id = self.env.agent_selection
            print(f"Terminations: {self.env.terminations}")
            
            step = 0
            truncated = False

            
            while not truncated and step <= self.max_steps:
                agent_id = self.env.agent_selection

                observation = self.env.observe(agent_id)
                
                # Choose action
                action, prob, val = self.choose_action(agent_id, observation)
                
                if self.env.terminations[agent_id]:
                    self.env.step(None)
                    continue
                # Take action in environment
                self.env.step(action)
                
                # Get reward and next observation
                reward = self.env.rewards[agent_id]
                done = self.env.terminations[agent_id] or self.env.truncations[agent_id]
                
                # Store experience
                self.memories[agent_id].store(
                    observation["observation"],
                    action,
                    prob,
                    val,
                    reward,
                    done,
                    observation["action_mask"]
                )
                
                episode_reward[agent_id] += reward
                
                # Check if it's time to learn
                if len(self.memories[agent_id].states) >= self.memories[agent_id].batch_size:
                    self.learn(agent_id)
                
                step += 1
                truncated = all(self.env.truncations.values())
            
            # Calculate average reward for this episode
            avg_episode_reward = sum(episode_reward.values()) / len(episode_reward)
            episode_rewards.append(avg_episode_reward)
            
            # Print training progress
            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
                
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    # Save best model
                    for agent_id in self.agents:
                        torch.save(self.agents[agent_id]["network"].state_dict(), f"best_model_agent_{agent_id}.pt")
                    
        return episode_rewards

def main():
    env = CatanEnv()
    
    # Init PPO agent
    ppo = MultiAgentPPO(
        env=env,
        hidden_dim=512,
        batch_size=32,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=4,
        max_steps=10000
    )
    
    # Train the agent
    n_episodes = 1000
    rewards = ppo.train(n_episodes)
    
    # Plot training rewards if desired
    print("Training completed!")
    print(f"Final average reward: {sum(rewards) / len(rewards):.2f}")

if __name__ == "__main__":
    main()
