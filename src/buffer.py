import numpy as np 
import torch
from types import SimpleNamespace

class ExpertBuffer:
    def __init__(self, cfg, env):
        # Paper: |M_k| = expert_buffer_size per task (default 10k).
        self.task_buffer_size = int(cfg.expert_buffer_size)
        self.task_list = cfg.task_list
        self.env = env
        self.device = cfg.device
        self.tasks = {}

    def add_observation_batch(self, observations, task_name, agent):
        """
        Add task observations from replay buffer to the expert buffer of size
        task_buffer_size. Random experience tuples are chosen and their target
        means and log stds are computed.

        Args:
            observations (numpy array): Observations to add.
            task_name (str): Task name to add.
            agent: Actor module that returns (mean, log_std).
        """
        obs_len = len(observations)
        if obs_len == 0:
            raise ValueError("Cannot add expert observations from an empty buffer")

        sample_size = min(self.task_buffer_size, obs_len)
        idx = np.random.choice(obs_len, size=sample_size, replace=False)
        observations = observations[idx]
        batch_size = 64

        means = []
        log_stds = []
        for start in range(0, len(observations), batch_size):
            end = min(start + batch_size, len(observations))
            with torch.no_grad():
                obs_batch = torch.as_tensor(
                    observations[start:end], dtype=torch.float32, device=self.device
                )
                mean, log_std = agent(obs_batch)
                means.append(mean.detach())
                log_stds.append(log_std.detach())

        mean_targets = torch.cat(means, dim=0)
        log_std_targets = torch.cat(log_stds, dim=0)
        observation = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        self.tasks[task_name] = [observation, mean_targets, log_std_targets]

    def sample_expert_batch(self, batch_size, task_name):
        buffer_size = len(self.tasks[task_name][0])
        idx = np.random.randint(buffer_size, size=batch_size)
        obs = self.tasks[task_name][0][idx]
        target_means = self.tasks[task_name][1][idx]
        target_log_stds = self.tasks[task_name][2][idx]
        return obs, target_means, target_log_stds


class ReplayBuffer:
    def __init__(self, cfg, env, capacity=None):
        self.replay_buffer_size = int(capacity if capacity is not None else cfg.replay_buffer_size)
        self.tasks = cfg.task_list
        self.env = env
        self.observations = np.zeros((self.replay_buffer_size, *self.env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((self.replay_buffer_size, *self.env.action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((self.replay_buffer_size,), dtype=np.float32)
        self.dones = np.zeros((self.replay_buffer_size,), dtype=np.float32)
        self.next_observations = np.zeros((self.replay_buffer_size, *self.env.observation_space.shape), dtype=np.float32)
        self.size = 0
        self.pointer = 0
        self.device = cfg.device
        self.num_envs = cfg.num_envs
    
    def add(self,observation, action, reward, done, next_observation):
    
        self.observations[self.pointer: self.pointer + self.num_envs] = observation
        self.actions[self.pointer: self.pointer + self.num_envs] = action
        self.rewards[self.pointer: self.pointer + self.num_envs] = reward
        self.dones[self.pointer: self.pointer + self.num_envs] = done
        self.next_observations[self.pointer: self.pointer + self.num_envs] = next_observation
        self.pointer = (self.pointer + self.num_envs) % self.replay_buffer_size
        self.size = min(self.size + self.num_envs, self.replay_buffer_size)
        
    
    def reset(self):
        self.size = 0
        self.pointer = 0

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return SimpleNamespace(
            observations=torch.tensor(self.observations[indices]).to(self.device),
            actions=torch.tensor(self.actions[indices]).to(self.device),
            rewards=torch.tensor(self.rewards[indices]).to(self.device),
            dones=torch.tensor(self.dones[indices]).to(self.device),
            next_observations=torch.tensor(self.next_observations[indices]).to(self.device),
        )
