import numpy as np 
import torch
from types import SimpleNamespace
class ExpertBuffer:
    def __init__(self, cfg, env, policy):
        self.expert_buffer_size = cfg.expert_buffer_size
        self.tasks = cfg.task_list
        self.env = env
        self.policy = policy
        self.device = cfg.device
        self.observations = np.zeros((self.expert_buffer_size, *self.env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((self.expert_buffer_size, *self.env.action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((self.expert_buffer_size,1), dtype=np.float32)
        self.dones = np.zeros((self.expert_buffer_size,1), dtype=np.float32)
        self.next_observations = np.zeros((self.expert_buffer_size, *self.env.observation_space.shape), dtype=np.float32)
        self.size = 0
        self.pointer = 0

        def add(self, observations, task):
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
                    # sample actions from observation batch
                    action_info = self.policy(observations[start:end], task)[1]
                    mean, log_std = action_info['mean'], action_info['log_std']
                    means.append(mean)
                    log_stds.append(log_std)
        
        mean_targets = torch.cat(means)
        log_std_targets = torch.cat(log_stds)

        if self._per_task:
            self.observation_buffer.append(observations)
            self.target_mean_buffer.append(mean_targets)
            self.target_log_std_buffer.append(log_std_targets)
        else:
            if self.observation_buffer is None:
                self.observation_buffer = observations
                self.target_mean_buffer = mean_targets
                self.target_log_std_buffer = log_std_targets
                self.task_idx_buffer = torch.tensor([seq_idx]*self._capacity_per_task)
            else:
                self.observation_buffer=torch.cat([self.observation_buffer, observations])
                self.target_mean_buffer = torch.cat([self.target_mean_buffer, mean_targets])
                self.target_log_std_buffer = torch.cat([self.target_log_std_buffer, log_std_targets])
                self.task_idx_buffer = torch.cat([self.task_idx_buffer, torch.tensor([seq_idx]*self._capacity_per_task)])

        self._capacity += self._capacity_per_task


        



class ReplayBuffer:
    def __init__(self, cfg, env):
        self.replay_buffer_size = cfg.replay_buffer_size
        self.tasks = cfg.task_list
        self.env = env
        self.observations = np.zeros((self.replay_buffer_size, *self.env.observation_space.shape), dtype=np.float32)
        self.actions = np.zeros((self.replay_buffer_size, *self.env.action_space.shape), dtype=np.float32)
        self.rewards = np.zeros((self.replay_buffer_size,1), dtype=np.float32)
        self.dones = np.zeros((self.replay_buffer_size,1), dtype=np.float32)
        self.next_observations = np.zeros((self.replay_buffer_size, *self.env.observation_space.shape), dtype=np.float32)
        self.size = 0
        self.pointer = 0
        self.device = cfg.device
    
    def add(self,observation, action, reward, done, next_observation):
        self.observations[self.pointer] = observation
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.next_observations[self.pointer] = next_observation
        self.pointer = (self.pointer + 1) % self.replay_buffer_size
        self.size = min(self.size + 1, self.replay_buffer_size)
        
    
    def reset(self):
        self.size = 0
        self.pointer = 0

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return SimpleNamespace(
            observations=torch.from_numpy(self.observations[indices]).to(self.device),
            actions=torch.from_numpy(self.actions[indices]).to(self.device),
            rewards=torch.from_numpy(self.rewards[indices]).to(self.device),
            dones=torch.from_numpy(self.dones[indices]).to(self.device),
            next_observations=torch.from_numpy(self.next_observations[indices]).to(self.device),
        )
         




        



