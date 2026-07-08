from .sac import SAC
from src.buffer import ExpertBuffer
import torch
from torch.distributions import Normal, Independent, kl_divergence
import numpy as np
from tqdm import tqdm
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
        self.target_policy = agent
        self.replay_buffer = buffer
        self.file_path = file_path
        self.task_name = task_name
        self.reset_offline_actor = cfg.reset_offline_actor
        self.expert_buffer = ExpertBuffer(cfg, env,self.task_name)
        self.rnd_batch_size = cfg.rnd_batch_size
        self.observations = self.replay_buffer.observations
        self.device = cfg.device
        self.results = {}
        self.results['Policy loss'] = []
        self.results['BC loss'] = []
        self.results['Speed (it/s)'] = []
        
    def compute_distill_setup(self):
        if self.offline_actor is None:
            self.offline_actor = self.target_agent
        # add the observations to the expert buffer
        self.expert_buffer.add_observation_batch(self.observations, self.task_name, self.target_agent)

    def load_target_policy(self, file_path):
        if self.target_agent is None:
            self.target_agent = SAC(self.cfg, self.env).reset()
            checkpoint = torch.load(file_path, map_location=self.device)
            self.target_agent.load_state_dict(checkpoint)
            print(f"Target policy loaded from {file_path}")
        else:
            self.target_agent = self.target_agent.actor
            print("target policy loaded from online policy")

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
                mean, log_std = self.target_agent(self.observations[start:end])
                means.append(mean)
                log_stds.append(log_std)
        means = torch.cat(means, dim=0)
        log_stds = torch.cat(log_stds, dim=0)
        return means, log_stds

    def bc_loss(self, means, log_stds, target_means, target_log_stds):
       # loop through all tasks
        for i in self.task
            means, 

    @staticmethod
    def gaussian_kl(dist1, dist2):
        mu1, log_std1 = dist1
        mu2, log_std2 = dist2
        p = Independent(Normal(mu1, log_std1.exp()), 1)
        q = Independent(Normal(mu2, log_std2.exp()), 1)
        return kl_divergence(p, q).mean()
    
    def train(self):
        self.compute_distill_setup(self)
        target_means, target_log_stds = self.compute_target(self)
        for epoch in range(self.nepochs_offline):
            
            idx_list = list(range(len(self.observations)))
            np.random.shuffle(idx_list)
            pbar = tqdm(
                range(0,len(self.observations), self.batch_size), 
                desc = 'Epoch '+str(epoch+1), 
                ascii = ' =',
                leave=True)

            for i in pbar:
                start = i
                end = start + self.batch_size
                if end > len(self.observations):
                    end = len(self.observations)
                idxs = idx_list[start:end]
                obs, target_means, target_log_stds = self.observations[idxs], target_means[idxs], target_log_stds[idxs]

                means, log_stds = self.offline_actor(obs)
                # first loss is the distillation loss
                distill_loss = self.gaussian_kl((target_means, target_log_stds), (means, log_stds))
                self.expert_buffer.sample_expert_batch(self.batch_size)
                bc_loss = self.gaussian_kl((means, log_stds), (target_means, target_log_stds)) * self.cl_reg_coef
                loss = distill_loss + bc_loss

                zero_optim_grads(self._policy_optimizer)
                loss.backward()
                self._policy_optimizer.step()

                end_time = time()

                global_step += 1

                if global_step % 1000 == 0:
                    if self._use_wandb:
                        wandb.log({
                            'Policy loss': policy_loss.item(),
                            'BC loss': bc_loss.item(),
                            'Speed (it/s)' : (self.global_step / (end_time - self.start_time))
                        })

                    self.results['Policy loss'].append(policy_loss.item())
                    self.results['BC loss'].append(bc_loss.item())
                    self.results['Speed (it/s)'].append((self.global_step / (end_time - self.start_time)))

            
        last_return = self._evaluate_policy(trainer.step_itr)
        self.save_results()
        
        self.on_task_start(seq_idx)
            