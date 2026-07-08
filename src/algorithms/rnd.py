from .sac import SAC
from src.buffer import ExpertBuffer
import torch
from copy import deepcopy
from torch.distributions import Normal, Independent, kl_divergence
import numpy as np
from tqdm import tqdm
from time import time
class RND_SAC(SAC):
    def __init__(self, cfg, env, task_name, buffer, agent=None, file_path=None):
        super().__init__(cfg, env)
        self.nepochs_offline = cfg.nepochs_offline
        self.batch_size=128
        self.cl_reg_coef = cfg.cl_reg_coef
        self.expert_buffer_size = cfg.expert_buffer_size
        self.replay_buffer_size = cfg.replay_buffer_size
        self.nepochs_offline = cfg.nepochs_offline
        self.env_seq = cfg.env_seq
        self.target_policy = agent.actor
        self.offline_policy = None
        self.replay_buffer = buffer
        self.file_path = file_path
        self.task_name = task_name
        self.cfg = cfg
        


        # self.reset_offline_policy = cfg.reset_offline_actor
        self.expert_buffer = ExpertBuffer(cfg, env,self.task_name)
        self.rnd_batch_size = cfg.rnd_batch_size
        self.observations = self.replay_buffer.observations
        self.device = cfg.device
        self.results = {'Policy loss': [],
                        'BC loss': [],
                        'Speed (it/s)': []}
        
        self.seen_tasks = []
        
    def compute_distill_setup(self):
        if self.file_path is not None:
            self.load_target_policy(self.file_path)

        self.target_policy = deepcopy(self.target_policy).to(self.device).eval()
        for param in self.target_policy.parameters():
            param.requires_grad_(False)

        if self.offline_policy is None:
            self.offline_policy = deepcopy(self.target_policy).to(self.device)
            self.offline_policy.train()
            self.offline_actor_optimizer = torch.optim.Adam(
                self.offline_policy.parameters(),
                lr=self.cfg.policy_lr,
            )

        # add the observations to the expert buffer
        self.expert_buffer.add_observation_batch(self.observations, self.task_name, self.target_policy)
        self.seen_tasks.append(self.task_name)

    def load_target_policy(self, file_path):
        
        target_agent = SAC(self.cfg, self.env).reset()
        checkpoint = torch.load(file_path, map_location=self.device)
        if "actor" in checkpoint:
            target_agent.actor.load_state_dict(checkpoint["actor"])
        else:
            target_agent.actor.load_state_dict(checkpoint)
        self.target_policy = target_agent.actor
        print(f"Target policy loaded from {file_path}")
    

    def compute_target(self):
        # we have to find a way to preserve the offline policy since
        #  we do not load the target policy as it is the current policy
        observations = torch.as_tensor(self.observations, dtype=torch.float32, device=self.device)
        means = []
        log_stds = []
        for start in range(0, len(observations), self.rnd_batch_size):
            end = min(start + self.rnd_batch_size, len(observations))
            with torch.no_grad():
                mean, log_std = self.target_policy(observations[start:end])
                means.append(mean.detach())
                log_stds.append(log_std.detach())
        return torch.cat(means, dim=0), torch.cat(log_stds, dim=0)

    def bc_loss(self):
        if len(self.seen_tasks) == 1:
            return torch.zeros(1).to(self.device)
        else:
            bc_loss = torch.zeros(1, device=self.device)
            for task_idx in range(len(self.seen_tasks)-1):
                obs, target_means, target_log_stds = self.expert_buffer.sample_expert_batch(self.batch_size, task_name= self.seen_tasks[task_idx])
                means, log_stds = self.offline_policy(obs)
                bc_loss += self.gaussian_kl((means, log_stds), (target_means, target_log_stds)) * self.cl_reg_coef

            return bc_loss

    @staticmethod
    def gaussian_kl(dist1, dist2):
        mu1, log_std1 = dist1
        mu2, log_std2 = dist2
        p = Independent(Normal(mu1, log_std1.exp()), 1)
        q = Independent(Normal(mu2, log_std2.exp()), 1)
        return kl_divergence(p, q).mean()
    
    def train(self):
        self.compute_distill_setup()
        all_target_means, all_target_log_stds = self.compute_target()
        observations = torch.as_tensor(self.observations, dtype=torch.float32, device=self.device)
        global_step = 0
        for epoch in range(self.nepochs_offline):
            
            idx_list = list(range(len(observations)))
            np.random.shuffle(idx_list)
            pbar = tqdm(
                range(0,len(observations), self.batch_size), 
                desc = 'Epoch '+str(epoch+1), 
                ascii = ' =',
                leave=True)

            for i in pbar:
                start = i
                end = start + self.batch_size
                if end > len(observations):
                    end = len(observations)
                idxs = idx_list[start:end]
                obs = observations[idxs]
                batch_target_means = all_target_means[idxs]
                batch_target_log_stds = all_target_log_stds[idxs]

                means, log_stds = self.offline_policy(obs)
                # first loss is the distillation loss
                distill_loss = self.gaussian_kl((batch_target_means, batch_target_log_stds), (means, log_stds))
                bc_loss = self.bc_loss()
                loss = distill_loss + bc_loss

                self.offline_actor_optimizer.zero_grad()
                loss.backward()
                self.offline_actor_optimizer.step()

                end_time = time()

                global_step += 1

                if global_step % 1000 == 0:
                    logging_cfg = getattr(self.cfg, "logging", None)
                    if logging_cfg is not None and bool(getattr(logging_cfg, "enable_wandb", False)):
                        import wandb

                        wandb.log({
                            "global_step": global_step,
                            "task_name": self.task_name,
                            "RND/Policy loss": distill_loss.item(),
                            "RND/BC loss": bc_loss.item(),
                            "RND/Speed (it/s)": global_step / (end_time - self.start_time),
                        })

                    self.results['Policy loss'].append(distill_loss.item())
                    self.results['BC loss'].append(bc_loss.item())
                    self.results['Speed (it/s)'].append((global_step / (end_time - self.start_time)))

        return self.offline_policy
            