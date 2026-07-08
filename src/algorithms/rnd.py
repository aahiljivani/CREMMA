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
        self.target_policy = agent.actor
        self.offline_policy = None
        self.replay_buffer = buffer
        self.file_path = file_path
        self.task_name = task_name
        # self.reset_offline_policy = cfg.reset_offline_actor
        self.expert_buffer = ExpertBuffer(cfg, env,self.task_name)
        self.rnd_batch_size = cfg.rnd_batch_size
        self.observations = self.replay_buffer.observations
        self.device = cfg.device
        self.results = {}
        self.results['Policy loss'] = []
        self.results['BC loss'] = []
        self.results['Speed (it/s)'] = []
        self.seen_tasks = []
        
    def compute_distill_setup(self):
        if self.offline_policy is None and self.file_path is None:
            self.offline_policy = self.target_policy
        elif self.file_path is not None:
            self.load_target_policy(self.file_path)

        # add the observations to the expert buffer
        self.expert_buffer.add_observation_batch(self.observations, self.task_name, self.target_policy)
        self.seen_tasks.append(self.task_name)

    def load_target_policy(self, file_path):
        
        self.target_policy = SAC(self.cfg, self.env).reset()
        checkpoint = torch.load(file_path, map_location=self.device)
        self.target_policy.load_state_dict(checkpoint)
        print(f"Target policy loaded from {file_path}")
    

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
       loss = 0
        for task_name in self.seen_tasks
                obs, target_means, target_log_stds = self._expert_buffer.sample_expert_batch(self.batch_size, task_name=task_name)
                means, log_stds = self.policy(obs)
                 

                if self._bc_kl == 'reverse':
                    loss += self.kl_loss((means, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef
                elif self._bc_kl == 'forward':
                    loss += self.kl_loss((target_means, target_log_stds),(means, log_stds)) * self._cl_reg_coef

                
        else:
            obs, target_means, target_log_stds, task_idxs = self._expert_buffer.sample_expert_batch(self.BATCH_SIZE)

            action_info = self.policy(obs, task_idxs)[1]
            means, log_stds = action_info['mean'], action_info['log_std']

            if self._bc_kl == 'reverse':
                loss = self.kl_loss((means, log_stds), (target_means, target_log_stds)) * self._cl_reg_coef
            elif self._bc_kl == 'forward':
                loss = self.kl_loss((target_means, target_log_stds),(means, log_stds)) * self._cl_reg_coef

        return loss 

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

                means, log_stds = self.offline_policy(obs)
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
            