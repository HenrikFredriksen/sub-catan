import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.distributions import Categorical
from collections import deque
import time
import os

from environment.CatanEnv_torch_spec import CatanEnv
from environment.CatanSettlePhaseEnv import CatanSettlePhaseEnv

class PPONetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PPONetwork, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        shared_features = torch.nn.functional.normalize(shared_features, dim=-1)

        policy_logits = self.policy_head[0](shared_features)
        masked_logits = policy_logits.masked_fill(action_mask == 0, float('-inf'))
        
        policy = torch.nn.functional.softmax(masked_logits, dim=-1)

        # Ensure policy is valid
        policy = torch.clamp(policy, 1e-7, 1)
        policy = policy / policy.sum(dim=-1, keepdim=True)
        
        value = self.value_head(shared_features)
        return policy, value
    
class ExlporativeNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=-1))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, 1))
        
    def forward(self, x, action_mask):
        shared_features = self.shared_layers(x)
        
        # Apply action mask to policy logits
        shared_features = torch.nn.functional.normalize(shared_features, dim=-1)

        policy_logits = self.policy_head[0](shared_features)
        masked_logits = policy_logits.masked_fill(action_mask == 0, float('-inf'))
        
        policy = torch.nn.functional.softmax(masked_logits, dim=-1)

        # Ensure policy is valid
        policy = torch.clamp(policy, 1e-7, 1)
        policy = policy / policy.sum(dim=-1, keepdim=True)
        
        value = self.value_head(shared_features)
        return policy, value
    
def get_policy_network(policy_type, input_dim, hidden_dim, output_dim):
    policies = {
        'baseline': PPONetwork,
        'explorative': ExlporativeNetwork
    }
    return policies[policy_type](input_dim, hidden_dim, output_dim)

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
        agent_policies=None,
        **kwargs
    ):
        self.env = env
        self.writer = writer
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.0003)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_epsilon = kwargs.get('clip_epsilon', 0.2)
        self.n_epochs = kwargs.get('n_epochs', 4)
        self.c1 = kwargs.get('c1', 0.5)
        self.c2 = kwargs.get('c2', 0.01)
        self.max_steps = kwargs.get('max_steps', 10000)
        
        # Init networks and optimizers for each agent
        self.agents = {}
        self.memories = {}
        
        agent_policies = agent_policies or {agent_id: 'baseline' for agent_id in env.possible_agents}
        
        for agent_id in self.env.possible_agents:
            obs_dim = env.observation_spaces[agent_id]["observation"].shape[0]
            act_dim = env.action_spaces[agent_id].n
            
            # Create network and optimizer
            
            
            network = get_policy_network(
                agent_policies[agent_id],
                obs_dim,
                kwargs.get('hidden_dim', 256),
                act_dim
            )
            optimizer = optim.Adam(network.parameters(), lr=kwargs.get('learning_rate', 0.0003))
            
            self.agents[agent_id] = {
                "network": network,
                "optimizer": optimizer,
                "policy_type": agent_policies[agent_id]
            }
            # Create memory
            self.memories[agent_id] = PPOMemory(kwargs.get('batch_size', 32))
    
    def choose_action(self, agent_id, observation):
        if self.env.terminations.get(agent_id, False):
            return None, None, None
         
        state = torch.FloatTensor(self.env.observe(agent_id)["observation"])
        action_mask = torch.FloatTensor(self.env.observe(agent_id)["action_mask"])
        
        with torch.no_grad():
            try:
                policy, value = self.agents[agent_id]["network"](state, action_mask)


                if torch.isnan(policy).any():
                    print(f"Warning: NaN in policy for agent {agent_id}")
                    # Return a dummy action
                    valid_actions = torch.nonzero(action_mask).flatten()
                    if len(valid_actions) == 0:
                        return None, None, None
                    action = valid_actions[torch.randint(0, len(valid_actions), (1,))]
                    return action.item(), 0.0, 0.0
            
                # Create distribution only for valid actions
                dist = Categorical(policy)
                action = dist.sample()
                prob = dist.log_prob(action)

                if action_mask[action] == 0:
                    # Fallback safe
                    valid_actions = torch.nonzero(action_mask).flatten()
                    action = valid_actions[torch.randint(0, len(valid_actions), (1,))]
                    prob = torch.log(torch.tensor(1.0 / len(valid_actions)))
       
                return action.item(), prob.item(), value.item()
            
            except Exception as e:
                print(f"Error choosing action for agent: {agent_id}: {str(e)}")
                valid_actions = torch.nonzero(action_mask).flatten()   
                if len(valid_actions) == 0:
                    return None, None, None
                action = valid_actions[torch.randint(0, len(valid_actions), (1,))]
                return action.item(), 0.0, 0.0
    
    def learn(self, agent_id, episode):

        policy_losses = []
        value_losses = []
        total_losses = []
        entropies = []

        try:
            memory = self.memories[agent_id]
            if len(memory.states) < self.batch_size:
                return

            network = self.agents[agent_id]["network"]
            optimizer = self.agents[agent_id]["optimizer"]

            states = torch.FloatTensor(np.array(memory.states))
            actions = torch.LongTensor(np.array(memory.actions))
            old_probs = torch.FloatTensor(np.array(memory.probs))
            vals = torch.FloatTensor(np.array(memory.vals))
            rewards = torch.FloatTensor(np.array(memory.rewards))
            dones = torch.FloatTensor(np.array(memory.dones))
            action_masks = torch.FloatTensor(np.array(memory.action_masks))
            
            advantages = torch.zeros_like(rewards)
            last_gae_lam = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_val = 0
                else:
                    next_val = vals[t + 1]
                    
                delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - vals[t]
                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lam
                
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

            for epoch in range(self.n_epochs):
                policy, value = network(states, action_masks)

                if torch.isnan(policy).any() or torch.isnan(value).any():
                    print(f"NaN detected in forward pass for agent {agent_id}")
                    break

                dist = Categorical(policy)
                new_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = (new_probs - old_probs).exp()
                ratio = torch.clamp(ratio, -10, 10)  # Prevent extreme ratios

                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
                
                policy_type = self.agents[agent_id]["policy_type"]
                
                entropy_coef = self.c2
                if policy_type == 'explorative':
                    entropy_coef = self.c2 * 2
                    
                    state_values = self.agents[agent_id]["network"].value_head(states)
                    state_std = state_values.std()
                    exploration_bonus = state_std * 0.1
                    advantages = advantages + exploration_bonus

                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = 0.5 * ((rewards - value) ** 2).mean()

                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Collect for logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                total_losses.append(loss.item())
                entropies.append(entropy.item())

                # Monitor loss values
                if torch.isnan(loss):
                    print(f"NaN loss detected for agent {agent_id}")
                    break

            memory.clear()

            avg_policy_loss = np.mean(policy_losses)
            avg_value_loss = np.mean(value_losses)
            avg_total_loss = np.mean(total_losses)
            avg_entropy = np.mean(entropies)
            
            self.writer.add_scalar(f"Policy Loss/Agent {agent_id}", avg_policy_loss, episode)
            self.writer.add_scalar(f"Value Loss/Agent {agent_id}", avg_value_loss, episode)
            self.writer.add_scalar(f"Total Loss/Agent {agent_id}", avg_total_loss, episode)
            self.writer.add_scalar(f"Entropy/Agent {agent_id}", avg_entropy, episode)

        except Exception as e:
            print(f"Error in learning step for agent {agent_id}: {e}")
            memory.clear()

    def train(self, n_episodes, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        best_reward = float('-inf')
        episode_rewards = []
        
        for episode in range(n_episodes):
            print(f"Starting episode {episode + 1}")
            
            episode_seed = seed + episode if seed is not None else None
            obs = self.env.reset(seed=episode_seed, return_info=True)[0]
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
            if (episode + 1) % 1 == 0:
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

        rankings = sorted(victory_points.items(), key=lambda x: x[1], reverse=True)

        position_scalars = {
            0: 1.5,
            1: 1.0,
            2: 0.5,
            3: 0.1
        }

        for position, (agent_id, vp) in enumerate(rankings):
            self.writer.add_scalar(f"Victory Points/{agent_id}", vp, episode)

            scalar = position_scalars[position]
            episode_reward[agent_id] *= scalar

            if position == 0 and self.env.game_manager.game_ended_by_victory_points:
                extra_reward = 20
                episode_reward[agent_id] += extra_reward
                print(f"Agent {agent_id} got extra reward of {extra_reward} for winning the game")
            
            print(f"{agent_id} placed {position + 1} with {vp} victory points and got reward {episode_reward[agent_id]}")

            vp_reward = vp * 2
            episode_reward[agent_id] += vp_reward

            if self.env.terminations.get(agent_id, False):
                termination_penalty = -10
                episode_reward[agent_id] += termination_penalty
                print(f"Agent {agent_id} got termination reward of {termination_penalty}")
            
            print(f"{agent_id} final reward: {episode_reward[agent_id]}")

def pretrain_settlement_phase():
    writer = SummaryWriter(log_dir='runs/catan_settle_pretraining')
    env = CatanSettlePhaseEnv(writer=writer)
    writer = SummaryWriter(log_dir='runs/catan_settle_pretraining')
    
    agent_policies = {
        'player_1': 'baseline',
        'player_2': 'explorative',
        'player_3': 'baseline',
        'player_4': 'explorative'
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
        max_steps=10000
    )

    n_episodes = 1500
    base_seed = 42
    rewards = ppo.train(n_episodes, seed=base_seed)

    for agent_id in ppo.agents:
        torch.save(
            ppo.agents[agent_id]["network"].state_dict(), 
            f"pretrained_settle_{agent_id}.pt"
        )

    writer.close()
    return ppo

def main():
    env = CatanEnv()
    
    writer = SummaryWriter(log_dir='runs/catan_training')
    
    agent_policies = {
        'player_1': 'baseline',
        'player_2': 'explorative',
        'player_3': 'baseline',
        'player_4': 'explorative'
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
        max_steps=10000
    )

    for agent_id in ppo.agents:
        pretrained_path = f"pretrained_settle_{agent_id}.pt"
        if os.path.exists(pretrained_path):
            ppo.agents[agent_id]["network"].load_state_dict(
                torch.load(pretrained_path)
            )
            print(f"Loaded pretrained model for agent {agent_id}")

    # Train the agent
    n_episodes = 1500
    base_seed = 42
    rewards = ppo.train(n_episodes, seed=base_seed)
    
    writer.close()
    
    # Plot training rewards if desired
    print("Training completed!")
    print(f"Final average reward: {sum(rewards) / len(rewards):.2f}")
    return ppo

def eval_trained_agents():
    env = CatanEnv(render_mode='human')
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
            if env.render_mode == 'human':
                env.render()
                
        except Exception as e:
            print(f"Error in evaluation loop: {e}")
            break

if __name__ == "__main__":
    #pretrain_settlement_phase()
    #main()
    eval_trained_agents()
    
