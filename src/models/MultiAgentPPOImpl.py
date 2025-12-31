import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

#profiling
from time import perf_counter

'''
This file contains the classes and functions for the MultiAgentPPO model implementation.
The MultiAgentPPO class is used to train multiple agents using the Proximal Policy Optimization (PPO) algorithm.
The PPONetwork class is the neural network model used by the agents to learn and make decisions.
The PPOMemory class is used to store experiences and sample batches for training.
The get_policy_network function is used to create the policy network for each agent, adding support for different policy types.

@Author: Henrik Tobias Fredriksen
@Date: 19. October 2024
'''
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
    
class AlternativeNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim * 2)),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, int(hidden_dim * 1.5)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim * 1.5), hidden_dim),
            nn.ReLU(),
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
        'alternative': AlternativeNetwork
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

        # global counter for "update steps" (one PPO epoch over one batch)
        self.global_step = 0
        self.update_step = 0
        
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
                 
        state = torch.FloatTensor(observation["observation"])
        action_mask = torch.FloatTensor(observation["action_mask"])
        
        with torch.no_grad():
            try:
                policy, value = self.agents[agent_id]["network"](state, action_mask)

                # Check for NaN in policy or value and set random valid_action if policy is NaN
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
                    print(f"Warning: Invalid action {action.item()} chosen for agent {agent_id}")
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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = perf_counter()

        policy_losses = []
        value_losses = []
        total_losses = []
        entropies = []

        try:
            memory = self.memories[agent_id]
            if len(memory.states) < self.batch_size:
                return 0.0

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

            for epoch in range(self.n_epochs):
                policy, value = network(states, action_masks)

                if torch.isnan(policy).any() or torch.isnan(value).any():
                    print(f"NaN detected in forward pass for agent {agent_id}")
                    break

                dist = Categorical(policy)
                new_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # 1. Approx‑KL (measured BEFORE we clamp ratio)
                approx_kl = (old_probs - new_probs).mean()

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

                grad_norm = torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

                optimizer.step()

                clip_fraction = (ratio.detach().abs() > self.clip_epsilon).float().mean()

                # Explained variance of the value function
                var_y = rewards.var(unbiased=False)
                explained_var = (torch.tensor(0.) if var_y == 0
                                 else 1 - (rewards - value.detach()).var(unbiased=False) / var_y)

                # Collect for logging
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                total_losses.append(loss.item())
                entropies.append(entropy.item())


                # push per‑epoch stats to TensorBoard
                self._log_update_stats(agent_id,
                                       policy_loss = policy_loss.item(),
                                       value_loss  = value_loss.item(),
                                       entropy     = entropy.item(),
                                       approx_kl   = approx_kl.item(),
                                       clip_frac   = clip_fraction.item(),
                                       grad_norm   = grad_norm.item(),
                                       explained_var = explained_var.item())


                # Monitor loss values
                if torch.isnan(loss):
                    print(f"NaN loss detected for agent {agent_id}")
                    break

            memory.clear()

            #avg_policy_loss = np.mean(policy_losses)
            #avg_value_loss = np.mean(value_losses)
            #avg_total_loss = np.mean(total_losses)
            #avg_entropy = np.mean(entropies)
            
            #self.writer.add_scalar(f"Policy Loss/Agent {agent_id}", avg_policy_loss, episode)
            #self.writer.add_scalar(f"Value Loss/Agent {agent_id}", avg_value_loss, episode)
            #self.writer.add_scalar(f"Total Loss/Agent {agent_id}", avg_total_loss, episode)
            #self.writer.add_scalar(f"Entropy/Agent {agent_id}", avg_entropy, episode)

        except Exception as e:
            print(f"Error in learning step for agent {agent_id}: {e}")
            memory.clear()
            return perf_counter() - t0

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return perf_counter() - t0

    def train(self, n_episodes, seed=None, max_env_calls=None, verbose=False, perf=False):
        stop_early = False

        if seed is not None:
            torch.manual_seed(seed)

        best_reward = float('-inf')
        episode_rewards = []

        
        # Create a directory for saved models if it doesn't exist
        os.makedirs("saved_models", exist_ok=True)

        # Determine the current run number once, at the start of training
        run_number = len([
            d for d in os.listdir("saved_models")
            if os.path.isdir(os.path.join("saved_models", d))
        ])

        if perf:
            save_dir_root = f"saved_models/run_{run_number+1:03}_perf"
        else:
            save_dir_root = f"saved_models/run_{run_number+1:03}"
        os.makedirs(save_dir_root, exist_ok=True)

        timing_csv = os.path.join(save_dir_root, "timing.csv")
        timing_fields = [
            "episode", 
            "env_calls", "act_calls", "obs_calls",
            "env_time_s_total", "env_ms_per_step",
            "act_time_s_total", "act_ms_per_call",
            "obs_time_s_total", "obs_ms_per_call",
            "learn_s_total",
            "episode_wall_s"
        ]
    
        #PROFILE
        run_wall_t0 = perf_counter()

        total_env_time = 0.0
        total_act_time = 0.0
        total_obs_time = 0.0
        total_learn_time = 0.0
        total_env_calls = 0
        total_act_calls = 0
        total_obs_calls = 0

        for episode in range(n_episodes):
            
            ep_wall_t0 = perf_counter()
            ep_env_time = 0.0
            ep_act_time = 0.0
            ep_obs_time = 0.0
            ep_learn_time = 0.0
            ep_env_calls = 0
            ep_act_calls = 0
            ep_obs_calls = 0
            
            episode_seed = (seed + episode) if seed is not None else None
            self.env.reset(seed=episode_seed, return_info=True)[0]

            done = {agent: False for agent in self.env.agents}
            episode_reward = {agent: 0 for agent in self.env.agents}
            
            step = 0

            while not all(done.values()):
                agent_id = self.env.agent_selection

                done[agent_id] = (
                    self.env.terminations.get(agent_id, False) or 
                    self.env.truncations.get(agent_id, False)
                )
                
                # Terminated / not active
                if done.get(agent_id, False) or agent_id not in self.env.agents:
                    t0 = perf_counter()
                    self.env.step(None)  # Pass None to step for terminated agents
                    dt = perf_counter() - t0
                    ep_env_time += dt
                    total_env_time += dt
                    ep_env_calls += 1
                    total_env_calls += 1

                    if max_env_calls is not None and total_env_calls >= max_env_calls:
                        stop_early = True
                        break
                    continue
                
                t0 = perf_counter()
                observation = self.env.observe(agent_id)
                dt = perf_counter() - t0
                ep_obs_time += dt
                total_obs_time += dt
                ep_obs_calls += 1
                total_obs_calls += 1

                # ------- action selection timing (policy forward + sampling)
                t0 = perf_counter()
                # Choose and take action
                action, prob, val = self.choose_action(agent_id, observation)
                dt = perf_counter() - t0
                
                ep_act_time += dt
                total_act_time += dt

                ep_act_calls += 1
                total_act_calls += 1

                if action is None:
                    # No valid action could be chosen
                    t0 = perf_counter()
                    self.env.step(None)
                    dt = perf_counter() - t0
                    ep_env_time += dt
                    total_env_time += dt
                    ep_env_calls += 1
                    total_env_calls += 1

                    if max_env_calls is not None and total_env_calls >= max_env_calls:
                        stop_early = True
                        break
                    continue

                # ------ env step timing
                t0 = perf_counter()
                # Take action in environment
                self.env.step(action)
                dt = perf_counter() - t0
                ep_env_time += dt
                total_env_time += dt
                ep_env_calls += 1
                total_env_calls += 1

                if max_env_calls is not None and total_env_calls >= max_env_calls:
                    stop_early = True
                    break

                # Update rewards and done status
                reward = self.env.rewards.get(agent_id, 0)
                #print(f"Agent {agent_id} took action {action} and got reward {reward}")
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

                if stop_early:
                    break

            if not stop_early:
                # After the episode ends, proceed to learning and reward calculation
                self.calculate_additional_rewards(episode_reward, episode, verbose)

                #logging
                total_episode_reward = sum(episode_reward.values())
                avg_episode_reward = sum(episode_reward.values()) / len(episode_reward)
                episode_rewards.append(avg_episode_reward)

                self._log_global(
                    AverageReward = avg_episode_reward,
                    TotalReward   = total_episode_reward,
                    EpisodeLength = self.env.step_count) # Duplicate values of step count in env and in trainer

                # Check if it's time to learn
                for agent_id in self.env.possible_agents:
                    self.writer.add_scalar(f"Agent_{agent_id}/Ep_Reward", episode_reward[agent_id], episode)
                    if len(self.memories[agent_id].states) > 0:
                        ep_learn_time += self.learn(agent_id, episode)


                # Calculate average reward for this episode
                for agent_id in self.env.possible_agents:
                    self.memories[agent_id].clear()

            ep_wall = perf_counter() - ep_wall_t0

            env_ms_per_step = 1000.0 * ep_env_time / max(1, ep_env_calls)
            act_ms_per_step = 1000.0 * ep_act_time / max(1, ep_act_calls)
            obs_ms_per_step = 1000.0 * ep_obs_time / max(1, ep_obs_calls)

            total_learn_time += ep_learn_time

            append_csv(timing_csv, timing_fields, {
                "episode": episode + 1,
                "env_calls": ep_env_calls,
                "act_calls": ep_act_calls,
                "obs_calls": ep_obs_calls,
                "env_time_s_total": ep_env_time,
                "env_ms_per_step": env_ms_per_step,
                "act_time_s_total": ep_act_time,
                "act_ms_per_call": act_ms_per_step,
                "obs_time_s_total": ep_obs_time,
                "obs_ms_per_call": obs_ms_per_step,
                "learn_s_total": ep_learn_time,
                "episode_wall_s": ep_wall
            })

            if stop_early:
                #print("Stopping early due to max_env_calls limit.")
                break

            # TensorBoard timing curves (trend view)
            self.writer.add_scalar("Timing/env_ms_per_step", env_ms_per_step, episode)
            self.writer.add_scalar("Timing/act_ms_per_step", act_ms_per_step, episode)
            self.writer.add_scalar("Timing/env_s_total", ep_env_time, episode)
            self.writer.add_scalar("Timing/act_s_total", ep_act_time, episode)
            self.writer.add_scalar("Timing/learn_s_total", ep_learn_time, episode)
            self.writer.add_scalar("Timing/episode_wall_s", ep_wall, episode)

            if (episode + 1) % 10 == 0:
                print(
                    f"[timing] ep={episode+1} "
                    f"env={env_ms_per_step:.3f} ms/step "
                    f"act={act_ms_per_step:.3f} ms/act "
                    f"learn={ep_learn_time:.3f} s "
                    f"wall={ep_wall:.3f} s"
                )
                        
            # Print training progress
            if (episode + 1) % 1 == 0:            
                avg_reward_all_episodes = sum(episode_rewards) / len(episode_rewards)
                print(f"Episode {episode + 1}/{n_episodes}: Steps: {step}, Average Reward: {avg_reward_all_episodes:.2f}")
                
                # Log the average reward for the episode, with 2 decimal places 
                #print(f"AVG REWARD: {episode_reward[agent_id]:.2f} | "
                #      f"P_1: {episode_reward['player_1']:.2f} | "
                #      f"P_2: {episode_reward['player_2']:.2f} | "
                #      f"P_3: {episode_reward['player_3']:.2f} | "
                #      f"P_4: {episode_reward['player_4']:.2f}")
                
                
                # Save models every 1000 episodes
                if (episode + 1) % 1000 == 0:
                    ep_dir = os.path.join(save_dir_root, f"ep_{episode + 1}")
                    os.makedirs(ep_dir, exist_ok=True)
                    for agent_id in self.agents:
                        model_path = os.path.join(ep_dir, f"model_agent_{agent_id}.pt")
                        torch.save(self.agents[agent_id]["network"].state_dict(), model_path)
                    print(f"Saved models for episode {episode + 1} in {ep_dir}")
                
                # If current average reward is better than all previous, save as "best_models"
                # TODO: Save only the best model of all agents instead of checking for avg change in last ep.
                # So, check if current iter of agent is better than earlier version of itself and save that model
                if (not perf) and avg_reward_all_episodes > best_reward:
                    best_reward = avg_reward_all_episodes
                    best_dir = os.path.join(save_dir_root, "best_models")
                    os.makedirs(best_dir, exist_ok=True)
                    for agent_id in self.agents:
                        model_path = os.path.join(best_dir, f"best_model_agent_{agent_id}.pt")
                        torch.save(self.agents[agent_id]["network"].state_dict(), model_path)
                    print(f"New best average reward {best_reward:.2f} — models saved to {best_dir}")

        run_wall = perf_counter() - run_wall_t0
        env_ms = 1000.0 * total_env_time / max(1, total_env_calls)
        act_ms = 1000.0 * total_act_time / max(1, total_act_calls)
        obs_ms = 1000.0 * total_obs_time / max(1, total_obs_calls)

        
        print(
            f"PERF_SUMMARY "
            f"env_calls={total_env_calls} "
            f"env_ms={env_ms:.3f} "
            f"act_calls={total_act_calls} "
            f"act_ms={act_ms:.3f} "
            f"obs_calls={total_obs_calls} "
            f"obs_ms={obs_ms:.3f} "
            f"learn_s={total_learn_time:.3f} "
            f"wall_s={run_wall:.3f}"
        )
    # Calculate rewards based on victory points and other game conditions
    def calculate_additional_rewards(self, episode_reward, episode, verbose):
        victory_points = self.env.get_victory_points()

        # Sort agents by victory points and reward based on placement
        rankings = sorted(victory_points.items(), key=lambda x: x[1], reverse=True)

        position_rewards = {
            0: 40,
            1: 25,
            2: 5,
            3: 0
        }

        for position, (agent_id, vp) in enumerate(rankings):
            self.writer.add_scalar(f"Agent_{agent_id}/VPs", vp, episode)

            scalar = position_rewards[position]
            episode_reward[agent_id] += scalar

            if position == 0 and self.env.game_manager.game_ended_by_victory_points:
                extra_reward = 20
                episode_reward[agent_id] += extra_reward
                if verbose:
                    print(f"Agent {agent_id} got extra reward of {extra_reward} for winning the game")

            if verbose:
                print(f"{agent_id} placed {position + 1} with {vp} victory points and got reward {episode_reward[agent_id]}")

            vp_reward = vp * 2
            episode_reward[agent_id] += vp_reward

            if self.env.terminations.get(agent_id, False):
                termination_penalty = -10
                episode_reward[agent_id] += termination_penalty
                if verbose:
                    print(f"Agent {agent_id} got termination penalty of {termination_penalty}")

    # Logging helper functions
    def _log_update_stats(self, agent_id: str, **scalars):
        for name, value in scalars.items():
            # e.g.  "Agent_player_1/policy_loss"
            tag = f"Agent_{agent_id}/{name}"
            self.writer.add_scalar(tag, value, self.update_step)
        # one log event = one "update step"
        self.update_step += 1

    def _log_global(self, **scalars):
        for name, value in scalars.items():
            self.writer.add_scalar(f"Global/{name}", value, self.global_step)
        self.global_step += 1          # create this counter in __init__

import csv

def append_csv(path, fieldnames, row_dict):
    file_exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerow(row_dict)
