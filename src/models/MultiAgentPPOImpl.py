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
@Date: 2. January 2026
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
        
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, action_mask):
        shared_features = self.shared_layers(x)

        # Apply action mask to policy logits
        policy_logits = self.policy_head(shared_features)
        masked_logits = policy_logits.masked_fill(action_mask == 0, -1e9)
        
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
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, action_mask):
        shared_features = self.shared_layers(x)
        
        # Apply action mask to policy logits
        policy_logits = self.policy_head(shared_features)
        masked_logits = policy_logits.masked_fill(action_mask == 0, -1e9)
        
        policy = torch.nn.functional.softmax(masked_logits, dim=-1)

        # Ensure policy is valid
        policy = torch.clamp(policy, 1e-7, 1.0)
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
    def __init__(self, batch_size, rng=None):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.batch_size = batch_size
        self.rng = rng if rng is not None else np.random.default_rng()
        
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
        self.rng.shuffle(indices)
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
        self.seed = kwargs.get('seed', 1234)
        self.rng = np.random.default_rng(self.seed)
        self.writer = writer
        self.batch_size = kwargs.get('batch_size', 32)
        self.rollout_steps = kwargs.get('rollout_steps', 1024)
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
    
    def choose_action(self, policy_id, observation):  
        state = torch.FloatTensor(observation["observation"])
        action_mask = torch.FloatTensor(observation["action_mask"])
        
        with torch.no_grad():
            policy, value = self.agents[policy_id]["network"](state, action_mask)
        
            # Check for NaN in policy or value and set random valid_action if policy is NaN
            if torch.isnan(policy).any():
                print(f"Warning: NaN in policy for agent {policy_id}")
                # Return a dummy action
                valid_actions = torch.nonzero(action_mask).flatten()
                if len(valid_actions) == 0:
                    return None, None, None
                action = valid_actions[torch.randint(0, len(valid_actions), (1,))]
                return action.item(), 0.0, 0.0
        
            # Create distribution only for valid actions
            dist = Categorical(policy)
            action = dist.sample()
            logp = dist.log_prob(action)
            if action_mask[action] == 0:
                print(f"Warning: Invalid action {action.item()} chosen for agent {policy_id}")
                # Fallback safe
                valid_actions = torch.nonzero(action_mask).flatten()
                action = valid_actions[torch.randint(0, len(valid_actions), (1,))]
                logp = torch.log(torch.tensor(1.0 / len(valid_actions)))
    
            return action.item(), logp.item(), value.item()
            
    def _seat_permutation(self, episode: int):
        seats = list(self.env.possible_agents)
        policies = list(self.env.possible_agents)
        perm = self.rng.permutation(policies)
        self.seat_to_policy = {seat: perm[i] for i, seat in enumerate(seats)}
    
    def learn(self, agent_id, episode):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = perf_counter()


        memory = self.memories[agent_id]
        T = len(memory.states)
        if T < self.batch_size:
            return 0.0
            
        network = self.agents[agent_id]["network"]
        optimizer = self.agents[agent_id]["optimizer"]

        # Tensor conversion
        states = torch.FloatTensor(np.array(memory.states))              # [T, obs]
        actions = torch.LongTensor(np.array(memory.actions))             # [T]
        old_logp = torch.FloatTensor(np.array(memory.probs))            # [T] (log probs)
        old_values = torch.FloatTensor(np.array(memory.vals))                  # [T]
        rewards = torch.FloatTensor(np.array(memory.rewards))            # [T]
        dones = torch.FloatTensor(np.array(memory.dones))                # [T] (dones 0/1)
        action_masks = torch.FloatTensor(np.array(memory.action_masks))  # [T, act_dim]
        
        with torch.no_grad():
            advantages = torch.zeros(T, dtype=torch.float32)
            last_gae = 0.0
            for t in reversed(range(T)):
                next_value = 0.0 if (t == T - 1) else old_values[t + 1]
                delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - old_values[t]
                last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
                advantages[t] = last_gae

        returns = advantages + old_values
        advantages_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_norm = advantages_norm.detach()
        returns = returns.detach()

        for epoch in range(self.n_epochs):
            idx = torch.randperm(T)
            nan_hit = False

            epoch_policy_losses = []
            epoch_value_losses = []
            epoch_entropies = []
            epoch_kls = []
            epoch_clip_fractions = []
            epoch_gradnorms = []

            for start in range(0, T, self.batch_size):
                mb = idx[start:start + self.batch_size]

                mb_states = states[mb]
                mb_masks = action_masks[mb].bool()
                mb_actions = actions[mb]
                mb_old_logp = old_logp[mb]
                mb_adv = advantages_norm[mb]
                mb_returns = returns[mb]

                policy, value = network(mb_states, mb_masks)
                value = value.squeeze(-1) # [B]

                dist = Categorical(policy)
                new_logp = dist.log_prob(mb_actions) # [B]
                entropy = dist.entropy().mean()

                log_ratio = new_logp - mb_old_logp
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

                surrogate1 = ratio * mb_adv
                surrogate2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = 0.5 * (mb_returns - value).pow(2).mean()

                loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
                # Monitor loss values
                if torch.isnan(loss):
                    print(f"NaN loss detected for agent {agent_id}")
                    nan_hit = True
                    break

                optimizer.zero_grad()
                loss.backward()
                grad_norm =torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()

                approx_kl = (mb_old_logp - new_logp).mean().item()
                clip_fraction = ((ratio.detach() - 1.0).abs() > self.clip_epsilon).float().mean().item()

                epoch_policy_losses.append(policy_loss.item())
                epoch_value_losses.append(value_loss.item())
                epoch_entropies.append(entropy.item())
                epoch_kls.append(approx_kl)
                epoch_clip_fractions.append(clip_fraction)
                epoch_gradnorms.append(float(grad_norm))

            if nan_hit or len(epoch_policy_losses) == 0:
                break
            # Logging per epoch
            with torch.no_grad():
                _, v_all = network(states, action_masks.bool())
                v_all = v_all.squeeze(-1)  # [T]
                var_y = returns.var(unbiased=False)
                explained_var = (torch.tensor(0.0) 
                                 if var_y < 1e-12 
                                 else 1.0 - (returns - v_all).var(unbiased=False) / var_y
                                 )
          
            self._log_update_stats(
                agent_id,
                policy_loss = float(np.mean(epoch_policy_losses)),
                value_loss = float(np.mean(epoch_value_losses)),
                entropy = float(np.mean(epoch_entropies)),
                approx_kl = float(np.mean(epoch_kls)),
                clip_frac = float(np.mean(epoch_clip_fractions)),
                grad_norm = float(np.mean(epoch_gradnorms)),
                explained_var = explained_var.item(),
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        memory.clear()
        return perf_counter() - t0

    def train(self, n_episodes, max_env_calls=None, verbose=False, perf=False):
        stop_early = False

        if self.seed is not None:
            torch.manual_seed(self.seed)

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

            
            #PROFILE
            ep_wall_t0 = perf_counter()
            ep_env_time = 0.0
            ep_act_time = 0.0
            ep_obs_time = 0.0
            ep_learn_time = 0.0
            ep_env_calls = 0
            ep_act_calls = 0
            ep_obs_calls = 0
            
            episode_seed = (self.seed + episode) if self.seed is not None else None
            self._seat_permutation(episode)
            obs, _ = self.env.reset(seed=episode_seed, return_info=True)

            done = {agent: False for agent in self.env.agents}
            episode_reward = {agent: 0 for agent in self.env.agents}
            policy_episode_reward = {pid: 0.0 for pid in self.env.possible_agents}
            
            step = 0
            last_idx = {}

            while not all(done.values()):
                seat_id = self.env.agent_selection

                done[seat_id] = (
                    self.env.terminations.get(seat_id, False) or 
                    self.env.truncations.get(seat_id, False)
                )
                
                # Terminated / not active
                if done.get(seat_id, False) or seat_id not in self.env.agents:
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
                observation = self.env.observe(seat_id)
                dt = perf_counter() - t0
                ep_obs_time += dt
                total_obs_time += dt
                ep_obs_calls += 1
                total_obs_calls += 1

                policy_id = self.seat_to_policy[seat_id]

                # ------- action selection timing (policy forward + sampling)
                t0 = perf_counter()
                # Choose and take action
                action, prob, val = self.choose_action(policy_id, observation)
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
                reward = self.env.rewards.get(seat_id, 0)
                policy_episode_reward[policy_id] += reward
                done[seat_id] = self.env.terminations.get(seat_id, False) or self.env.truncations.get(seat_id, False)
                episode_reward[seat_id] += reward

                # Store experience
                mem = self.memories[policy_id]
                mem.store(
                    state=observation['observation'],
                    action=action,
                    prob=prob,
                    val=val,
                    reward=reward,
                    done=done[seat_id],
                    action_mask=observation['action_mask']
                )
                last_idx[policy_id] = len(mem.rewards) - 1
                
                step += 1

                if stop_early:
                    break

            if not stop_early:
                for pid, i in last_idx.items():
                    self.memories[pid].dones[i] = 1.0  # Mark last step as done
                # After the episode ends, proceed to learning and reward calculation
                seat_bonus = self.calculate_additional_rewards(episode)
                for seat_id, bonus in seat_bonus.items():
                    pid = self.seat_to_policy[seat_id]
                    if pid in last_idx:
                        self.memories[pid].rewards[last_idx[pid]] += bonus
                        policy_episode_reward[pid] += bonus

                #logging
                total_episode_reward = sum(episode_reward.values())
                avg_episode_reward = sum(episode_reward.values()) / len(episode_reward)
                episode_rewards.append(avg_episode_reward)

                self._log_global(
                    AverageReward = avg_episode_reward,
                    TotalReward   = total_episode_reward,
                    EpisodeLength = self.env.step_count) # Duplicate values of step count in env and in trainer

                # Check if it's time to learn
                for policy_id in self.env.possible_agents:
                    self.writer.add_scalar(f"Agent_{policy_id}/Ep_Reward", policy_episode_reward[policy_id], episode)
                    if len(self.memories[policy_id].states) >= self.rollout_steps:
                        ep_learn_time += self.learn(policy_id, episode)

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
                        
            if episode % 50 == 0:
                print(policy_id, "buffer_len", len(self.memories[policy_id].states), "updates", self.update_step)

            # Print training progress
            if (episode + 1) % 1 == 0:            
                avg_episode_reward = sum(episode_rewards) / len(episode_rewards)
                #
                print(f"Episode {episode + 1}/{n_episodes}: Steps: {step}, Average Reward: {avg_episode_reward:.2f}")
                
                # Save models every 1000 episodes
                if (episode + 1) % 1000 == 0:
                    ep_dir = os.path.join(save_dir_root, f"ep_{episode + 1}")
                    os.makedirs(ep_dir, exist_ok=True)
                    for policy_id in self.agents:
                        model_path = os.path.join(ep_dir, f"model_agent_{policy_id}.pt")
                        torch.save(self.agents[policy_id]["network"].state_dict(), model_path)
                    print(f"Saved models for episode {episode + 1} in {ep_dir}")
                
                # If current average reward is better than all previous, save as "best_models"
                # TODO: Save only the best model of all agents instead of checking for avg change in last ep.
                # So, check if current iter of agent is better than earlier version of itself and save that model
                if (not perf) and avg_episode_reward > best_reward:
                    best_reward = avg_episode_reward
                    best_dir = os.path.join(save_dir_root, "best_models")
                    os.makedirs(best_dir, exist_ok=True)
                    for policy_id in self.agents:
                        model_path = os.path.join(best_dir, f"best_model_agent_{policy_id}.pt")
                        torch.save(self.agents[policy_id]["network"].state_dict(), model_path)
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
    def calculate_additional_rewards(self, episode):
        victory_points = self.env.get_victory_points()

        # Sort agents by victory points and reward based on placement
        rankings = sorted(victory_points.items(), key=lambda x: x[1], reverse=True)

        position_rewards = [40, 25, 5, -20]
        seat_bonus = {seat: 0.0 for seat in victory_points}

        for position, (seat_id, vp) in enumerate(rankings):
            self.writer.add_scalar(f"Agent_{seat_id}/VPs", vp, episode)

            pid = self.seat_to_policy[seat_id]
            self.writer.add_scalar(f"Agent_{pid}/VPs", vp, episode)

            seat_bonus[seat_id] += position_rewards[position]
            seat_bonus[seat_id] += 2.0 * vp  # 2 reward per victory point

            if position == 0 and self.env.game_manager.game_ended_by_victory_points:
                seat_bonus[seat_id] += 20.0  # Bonus for winning by VPs

            if self.env.terminations.get(seat_id, False):
                seat_bonus[seat_id] -= 10.0  # Penalty for elimination

        return seat_bonus
            

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
