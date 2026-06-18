import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def _single_observation_space(env):
    return env.observation_space


def _single_action_space(env):
    
    return env.action_space

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        observation_space = _single_observation_space(env)
        action_space = _single_action_space(env)
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod() + np.prod(action_space.shape), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, hidden_size):
        self.hidden_size = hidden_size
        super().__init__()
        observation_space = _single_observation_space(env)
        action_space = _single_action_space(env)
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_mu = nn.Linear(self.hidden_size, np.prod(action_space.shape)) # deterministic policy
        # action rescaling so we can clip the action to the bounds
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class DDPG:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = cfg.device
        self.hidden_size = cfg.hidden_size
        self.learning_rate = cfg.learning_rate
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.buffer_size = cfg.buffer_size
        self.learning_starts = cfg.learning_starts
        self.policy_frequency = cfg.policy_frequency
        self.start_time = time.time()


    def reset(self):
        self.actor = Actor(self.env, self.hidden_size).to(self.device)
        self.qf1 = QNetwork(self.env, self.hidden_size).to(self.device)
        self.qf1_target = QNetwork(self.env, self.hidden_size).to(self.device)
        self.target_actor = Actor(self.env, self.hidden_size).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()), lr=self.learning_rate)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.learning_rate)
        return self
    
    def predict(self, obs):
        with torch.no_grad():
                actions = self.actor(torch.Tensor(obs).to(self.device))
                actions += torch.normal(0, self.actor.action_scale * self.cfg.exploration_noise)
                action_space = _single_action_space(self.env)
                actions = actions.cpu().numpy().clip(action_space.low, action_space.high)
                return actions

    def update(self, data, global_step):
        actor_loss = None
        with torch.no_grad():
            next_state_actions = self.target_actor(data.next_observations)
            qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
            terminated = getattr(data, "terminated", None)
            if terminated is None:
                terminated = data.dones
            next_q_value = data.rewards.flatten() + (1 - terminated.flatten()) * self.gamma * (qf1_next_target).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
        self.q_optimizer.zero_grad()
        qf1_loss.backward()
        self.q_optimizer.step()

        if global_step % self.policy_frequency == 0:
            actor_loss = -self.qf1(data.observations, self.actor(data.observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if global_step % 100 != 0:
            return {}

        elapsed_time = max(time.time() - self.start_time, 1e-9)
        metrics = {
            "losses/qf1_values": qf1_a_values.mean().item(),
            "losses/qf1_loss": qf1_loss.item(),
            "losses/actor_loss": actor_loss.item() if actor_loss is not None else None,
            "charts/SPS": int(global_step / elapsed_time),
        }
        return metrics

    def save(self, path):
        model_path = f"{path}/{self.cfg.exp_name}.cleanrl_model"
        torch.save((self.actor.state_dict(), self.qf1.state_dict()), model_path)
        print(f"model saved to {model_path}")
        return model_path
