from .sac import SAC
from src.buffer import ExpertBuffer
import torch
from torch.distributions import Normal, Independent, kl_divergence

class RND_SAC(SAC):
    def __init__(self, cfg, env, task_name, agent, buffer):
        super().__init__(cfg, env)
        self.cl_reg_coef = cfg.cl_reg_coef
        self.expert_buffer_size = cfg.expert_buffer_size
        self.replay_buffer_size = cfg.replay_buffer_size
        self.nepochs_offline = cfg.nepochs_offline
        self.env_seq = cfg.env_seq
        self.agent = agent
        self.replay_buffer = buffer
        self.bc_kl = 'reverse'
        self.distill_kl = 'forward'
        self.task_name = task_name
        self.reset_offline_actor = cfg.reset_offline_actor
        self.expert_buffer = ExpertBuffer(cfg, env,self.task_name, self.agent)
        self.rnd_batch_size = cfg.rnd_batch_size
        self.observations = self.replay_buffer.observations
        
        self.results = {}
        self.results['Policy loss'] = []
        self.results['BC loss'] = []
        self.results['Speed (it/s)'] = []
        

    def compute_target(self):
        # we have to find a way to preserve the offline policy since
        #  we do not load the target policy as it is the current policy
        means = []
        log_stds = []
        for i in range(0,len(self.observations),self.batch_size):
            start = i
            if i+self.rnd_batch_size > len(self.observations):
                end = len(self.observations)
            else:
                end = i+self.rnd_batch_size
            with torch.no_grad():
                mean, log_std = self.agent.actor(self.observations[start:end])
                means.append(mean)
                log_stds.append(log_std)
        means = torch.cat(means, dim=0)
        log_stds = torch.cat(log_stds, dim=0)
        return means, log_stds

    @staticmethod
    def gaussian_kl(dist1, dist2):
        mu1, log_std1 = dist1
        mu2, log_std2 = dist2
        p = Independent(Normal(mu1, log_std1.exp()), 1)
        q = Independent(Normal(mu2, log_std2.exp()), 1)
        return kl_divergence(p, q).mean()
    
    def train(self):
        for epoch in range(self.nepochs_offline):
            means, log_probs = self.compute_target()
            self.agent.train(means, log_probs)
    