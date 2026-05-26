from nt import replace
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from continual_bench.envs import ContinualBenchEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from .cfg import parse_cfg
from .logger import ContinualLogger
from src.algorithms import RandomPolicy
import torch
from cleanrl_utils.buffers import ReplayBuffer


class ContinualBenchVecEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = int(cfg.seed)
        self.num_envs = int(cfg.num_envs)
        self.task_list = list(cfg.task_list)
        self.benchmark_mode = str(cfg.benchmark_mode)
        self.single_task_name = cfg.get("single_task_name", None)
        self.vec_env_cls = self._resolve_vec_env_cls(cfg.vec_env_cls)
        self.train_episodes_per_task = int(cfg.train.episodes_per_task)
        self.train_timesteps_per_episode = int(cfg.train.timesteps_per_episode)
        self.eval.eval_every_steps = int(cfg.eval.eval_every_steps)
        self.eval.num_eval_episodes = int(cfg.eval.num_eval_episodes)

    @staticmethod
    def _resolve_vec_env_cls(vec_env_name: str):
        if vec_env_name == "SubprocVecEnv":
            return SubprocVecEnv
        if vec_env_name == "DummyVecEnv":
            return DummyVecEnv
        raise ValueError("vec_env_cls must be one of {'SubprocVecEnv', 'DummyVecEnv'}")

    def _make_single_env(self, rank: int, task_name: str):
        ''' 
        creates a single environment for task_name with unique
        seed and gymnasium compatibility wrapper
        '''
        def _init():
            env = ContinualBenchEnv(render_mode=None, seed=self.seed + rank)
            env.set_task(task_name)
            return GymV21CompatibilityV0(env=env)
        return _init

    def _build_training_order(self) -> List[str]:
        '''
        error handling  and shuffling for random task order
        '''
        if self.benchmark_mode == "continual":
            return list(self.task_list)

        if self.benchmark_mode == "random":
            tasks = list(self.task_list)
            rng = random.Random(self.seed)
            rng.shuffle(tasks)
            return tasks # TODO we need to shuffle 720 times to get all permutations. Future work.

        if self.benchmark_mode == "task":
            if self.single_task_name not in self.task_list:
                raise ValueError(
                    f"single_task_name={self.single_task_name} must be one of {self.task_list}"
                )
            return [self.single_task_name]

        raise ValueError(
            f"Unknown benchmark_mode={self.benchmark_mode}. "
            "Expected one of {'continual', 'random', 'task'}"
        )

    def make_envs(self) -> Dict[str, SubprocVecEnv]: 
        ''' 
        creates vectorized environment for parallel envs training across single task.
        returns a dict mapping task_name to corresponding vectorized env.
         '''
        vec_envs = {}
        for task_name in self._build_training_order():
            env_fns = [self._make_single_env(i, task_name) for i in range(self.num_envs)]
            vec_envs[task_name] = self.vec_env_cls(env_fns, start_method="spawn")
        return vec_envs

    def _build_policy(self, action_space, num_envs: int): # building SAC and PPO policies soon.
        if self.cfg.policy == "RandomPolicy":
            return RandomPolicy(action_space, num_envs)
        if self.cfg.policy == "SACPolicy":
            return SACPolicy(action_space, num_envs)
        raise ValueError(f"Unsupported policy={self.cfg.policy}")

    def evaluate_seen_tasks(self, seen_tasks: List[str]) -> Dict[str, float]:
        task_scores = {}

        for task_name in seen_tasks:
            eval_env = DummyVecEnv([self._make_single_env(0, task_name)])
            eval_policy = self._build_policy(eval_env.action_space, num_envs=1)
            episode_scores = []

            for _ in range(int(self.cfg.eval.num_eval_episodes)):
                obs = eval_env.reset()
                done = [False]
                step_scores = []

                while not done[0]:
                    actions = eval_policy.predict(obs, deterministic=True)
                    obs, rewards, done, infos = eval_env.step(actions)
                    step_scores.append(float(infos[0][task_name]["success"]))

                episode_scores.append(float(np.mean(step_scores)) if step_scores else 0.0)

            eval_env.close()
            task_scores[task_name] = float(np.mean(episode_scores)) if episode_scores else 0.0

        return task_scores

    def train(self):
        # seeding
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = self.cfg.torch_deterministic
        # device
        self.cfg.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.cuda else "cpu")
        print(f"Using device: {self.cfg.device}")
        # create the vectorized environments
        vec_envs = self.make_envs()
        training_order = list(vec_envs.keys())
        # we need to change this, where we have to feed in every tasks vec env not just the first task
        first_env = vec_envs[training_order[0]]
        # building the policy will come from intitializing the agent first. i.e. calling SAC.reset()
        policy = self._build_policy(first_env.action_space, num_envs=self.num_envs)
        agent = policy.reset()
        if self.cfg.replay_buffer:
            rb = ReplayBuffer(
                self.cfg.buffer_size,
                first_env.observation_space,
                first_env.action_space,
                self.cfg.device,
                n_envs=self.num_envs,
                handle_timeout_termination=False,
            )

        # logging CRL specific metrics here.
        run_name = f"{self.cfg.policy}_{self.cfg.benchmark_mode}_{self.num_envs}env_seed{self.seed}"
        logger = ContinualLogger(
            project=self.cfg.logging.project,
            run_name=run_name,
            enable_wandb=bool(self.cfg.logging.enable_wandb),
            entity=self.cfg.logging.get("wandb_entity", None),
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

        # defining which tasks we have already seen
        seen_tasks = []
        # for the task index and task name
        for task_idx, task_name in enumerate(training_order):
            seen_tasks.append(task_name)
            vec_env = vec_envs[task_name]
            # start with the vectorized env of task

            for ep in range(int(self.train_episodes_per_task)):
                global_step = 0
                obs = vec_env.reset() # reset the vec env

                for t in range(int(self.train_timesteps_per_episode)):
                    global_step += 1
                    if global_step < self.cfg.learning_starts:
                        actions = np.array([vec_env.single_action_space.sample() for _ in range(vec_env.num_envs)])
                    else:
                    # we can specify updates here and task specific buffer stuff. like if policy is sac warmup buffer and append to buffer
                    # HERE
                        actions = agent.predict(obs, deterministic=False)
                    
                    obs, rewards, dones, infos = vec_env.step(actions)

                    # more logging of CRL metrics
                    successes = np.array(
                        [float(info[task_name]["success"]) for info in infos],
                        dtype=np.float32,
                    )

                    logger.update_online(
                        task_name=task_name,
                        task_idx=task_idx,
                        episode_idx=ep,
                        timestep_in_episode=t,
                        successes=successes,
                    )
                    # here is where we evaluate all the seen tasks
                    if logger.global_step % int(self.cfg.eval.eval_every_steps) == 0:
                        task_scores = self.evaluate_seen_tasks(seen_tasks)
                        logger.log_evaluation(seen_tasks, task_scores)

            vec_env.close()
        # finish eval and logging
        final_scores = self.evaluate_seen_tasks(seen_tasks)
        logger.log_evaluation(seen_tasks, final_scores)
        logger.finish(final_scores)


def main():
    cfg_dir = Path(__file__).resolve().parent.parent / "cfgs"
    cfg = parse_cfg(cfg_dir)
    print(OmegaConf.to_yaml(cfg))
    vecenv = ContinualBenchVecEnv(cfg)
    vecenv.train()


if __name__ == "__main__":
    main()