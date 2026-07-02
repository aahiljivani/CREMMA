import numpy as np 
import torch
from types import SimpleNamespace

class ExpertBuffer:
    def __init__(self, cfg, env, agent, task_name):
        self.expert_buffer_size = cfg.expert_buffer_size
        self.task_list = cfg.task_list
        self.env = env
        self.device = cfg.device
        self.observation_buffer = []
        self.target_mean_buffer = []
        self.target_log_std_buffer = []
        self.tasks = []
        self.task_buffer_size = self.expert_buffer_size// len(self.task_list)
        self.agent = agent

    def add_observation_batch(self, observations, task_name):
        """
        Add a episodic batch to the buffer.

        Args:
            episodes (EpisodeBatch): Episodes to add.
            observations are a numpy array

        """
        
        assert len(observations) == self.task_buffer_size

        
        BATCH_SIZE = 64
        
        means = []
        log_stds = []

        for i in range(0,len(observations),BATCH_SIZE):
            start = i
            if i+BATCH_SIZE > len(observations):
                end = len(observations)
            else:
                end = i+BATCH_SIZE
            with torch.no_grad():
                mean, log_std = self.agent.actor.forward(torch.tensor(observations[start:end]).to(self.device))
                means.append(mean)
                log_stds.append(log_std)
        
        mean_targets = torch.cat(means)
        log_std_targets = torch.cat(log_stds)

        if len(self.tasks)>=1:
            observations = torch.tensor(observations).to(self.device)
            self.observation_buffer = torch.cat([self.observation_buffer, observations])
            self.target_mean_buffer = torch.cat([self.target_mean_buffer, mean_targets])
            self.target_log_std_buffer = torch.cat([self.target_log_std_buffer, log_std_targets])
        else:
            self.observation_buffer = torch.tensor(observations).to(self.device)
            self.target_mean_buffer = mean_targets
            self.target_log_std_buffer = log_std_targets
        
        self.tasks.append(task_name)

    def sample_expert_batch(self, batch_size):
        buffer_size = len(self.tasks) * self.task_buffer_size
        idx = np.random.randint(buffer_size, size=batch_size)
        obs = self.observation_buffer[idx]
        target_means = self.target_mean_buffer[idx]
        target_log_stds = self.target_log_std_buffer[idx]
        
        return obs, target_means, target_log_stds
    


class ReplayBuffer:
    def __init__(self, cfg, env):
        self.replay_buffer_size = cfg.replay_buffer_size
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

         




        



