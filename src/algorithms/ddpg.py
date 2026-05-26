import os
import random
import time
from dataclasses import dataclass
from tkinter import HIDDEN
from turtle import hideturtle

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, hidden_size):
        self.hidden_size = hidden_size
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), hidden_size)
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
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_mu = nn.Linear(self.hidden_size, np.prod(env.single_action_space.shape)) # deterministic policy
        # action rescaling so we can clip the action to the bounds
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def reset(Actor, QNetwork, cfg):
    actor = Actor(envs, cfg.hidden_size).to(device)
    qf1 = QNetwork(envs, args.hidden_size).to(device)
    qf1_target = QNetwork(envs, args.hidden_size).to(device)
    target_actor = Actor(envs, args.hidden_size).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=cfg.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=cfg.learning_rate)