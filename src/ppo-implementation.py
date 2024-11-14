import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
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
        writer,
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
        self.writer = writer
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
    
    def learn(self, agent_id, episode):
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
        
        policy_losses = []
        value_losses = []
        total_losses = []
        entropies = []
                
        
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
                
                # Collect for logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                total_losses.append(loss.item())
                entropies.append(entropy.item())
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_total_loss = np.mean(total_losses)
        avg_entropy = np.mean(entropies)
        
        self.writer.add_scalar(f"Policy Loss/Agent {agent_id}", avg_policy_loss, episode)
        self.writer.add_scalar(f"Value Loss/Agent {agent_id}", avg_value_loss, episode)
        self.writer.add_scalar(f"Total Loss/Agent {agent_id}", avg_total_loss, episode)
        self.writer.add_scalar(f"Entropy/Agent {agent_id}", avg_entropy, episode)
                
                
        memory.clear()

    def train(self, n_episodes):
        best_reward = float('-inf')
        episode_rewards = []
        
        for episode in range(n_episodes):
            print(f"Starting episode {episode + 1}")
            obs = self.env.reset()
            done = {agent: False for agent in self.env.agents}
            episode_reward = {agent: 0 for agent in self.env.agents}
            
            step = 0
            
            while not all(done.values()):
                agent_id = self.env.agent_selection

                done[agent_id] = (
                self.env.terminations.get(agent_id, False) or 
                self.env.truncations.get(agent_id, False)
                )
                
                
                # Skip agents that have been terminated
                if done.get(agent_id, False) or agent_id not in self.env.agents:
                    self.env.step(None)  # Pass None to step for terminated agents
                    continue
                
                observation = self.env.observe(agent_id)

                # Choose and take action
                action_data = self.choose_action(agent_id, observation)
                if action_data is None:
                    # If no valid action, pass
                    self.env.step(None)
                    continue
                else:
                    action, prob, val = action_data

                # Take action in environment
                self.env.step(action)

                # Update rewards and done status
                reward = self.env.rewards.get(agent_id, 0)
                print(f"Agent {agent_id} took action {action} and got reward {reward}")
                done[agent_id] = self.env.terminations.get(agent_id, False) or self.env.truncations.get(agent_id, False)
                episode_reward[agent_id] += reward

                # Store experience
                self.memories[agent_id].store(
                    state=observation['observation'],
                    action=action,
                    prob=prob,
                    val=val,
                    reward=reward,
                    done=done[agent_id],
                    action_mask=observation['action_mask']
                )
                
                step += 1

            # After the episode ends, proceed to learning and reward calculation
            self.calculate_rewards(episode_reward, episode)
            
            #logging
            total_episode_reward = sum(episode_reward.values())
            avg_episode_reward = sum(episode_reward.values()) / len(episode_reward)
            episode_rewards.append(avg_episode_reward)
            
            self.writer.add_scalar("Average Reward per episode", avg_episode_reward, episode)
            
            # Check if it's time to learn
            for agent_id in self.env.possible_agents:
                if len(self.memories[agent_id].states) > 0:
                    self.learn(agent_id, episode)
                        
            # Calculate average reward for this episode
            
            for agent_id in self.env.possible_agents:
                self.memories[agent_id].clear()
            
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

    def calculate_rewards(self, episode_reward, episode):
        victory_points = self.env.get_victory_points()

        winner = None
        max_points = 10
        for agent_id, vp in victory_points.items():
            self.writer.add_scalar(f"Victory Points/{agent_id}", vp, episode)
            if vp > max_points:
                max_points = vp
                winner = agent_id

        for agent_id in self.env.possible_agents:
            if agent_id == winner and self.env.game_manager.check_if_game_ended():
                extra_reward = 20
                episode_reward[agent_id] += extra_reward
                print(f"Agent {agent_id} won the game with {max_points} victory points. Extra reward: {extra_reward}")

            vp_reward = victory_points[agent_id]
            episode_reward[agent_id] += vp_reward
            print(f"Agent {agent_id} got {vp_reward} victory points this episode")

            if self.env.terminations.get(agent_id, False):
                penalty = -10
                episode_reward[agent_id] += penalty
                print(f"Agent {agent_id} terminated early. Penalty {penalty}, total reward: {episode_reward[agent_id]}")
            else:
                print(f"Agent {agent_id} finished the episode with reward {episode_reward[agent_id]}")

def main():
    env = CatanEnv()
    
    writer = SummaryWriter(log_dir='runs/catan_training')
    
    # Init PPO agent
    ppo = MultiAgentPPO(
        env=env,
        writer=writer,
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
    n_episodes = 10
    rewards = ppo.train(n_episodes)
    
    writer.close()
    
    # Plot training rewards if desired
    print("Training completed!")
    print(f"Final average reward: {sum(rewards) / len(rewards):.2f}")

if __name__ == "__main__":
    main()
