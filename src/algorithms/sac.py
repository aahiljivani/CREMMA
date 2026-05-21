class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

class SAC:
    def __init__(self, cfg):
        self.actor = Actor(cfg.env)
        self.qf1 = SoftQNetwork(env)
        self.qf2 = SoftQNetwork(env)
        self.qf1_target = SoftQNetwork(env)
        self.qf2_target = SoftQNetwork(env)

    def predict(self, x, deterministic = False):
        mean, log_std = self(x)
        std = log_std.exp()
        if deterministic:
            mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
            return mean_action, None, None # consistent tuple shape
        else:

            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def reset(self):
        # start/boot all networks and 
        actor = Actor(envs).to(device)
        qf1 = SoftQNetwork(envs).to(device)
        qf2 = SoftQNetwork(envs).to(device)
        qf1_target = SoftQNetwork(envs).to(device)
        qf2_target = SoftQNetwork(envs).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

        # Automatic entropy tuning
        if cfg.autotune:
            target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        else:
            alpha = args.alpha

        envs.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.num_envs,
            handle_timeout_termination=False,
        )
    

    def update(self):
        pass
