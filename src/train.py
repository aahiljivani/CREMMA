import os

# Select an offscreen OpenGL backend BEFORE any mujoco import so eval rendering
# works on headless servers (no DISPLAY). EGL uses the GPU; set MUJOCO_GL=osmesa
# for a CPU fallback if EGL is unavailable. setdefault respects a user override.
os.environ.setdefault("MUJOCO_GL", "egl")

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
from src.algorithms import DDPG, RandomPolicy, SAC
import torch
from stable_baselines3.common.buffers import ReplayBuffer
import torch.optim as optim



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
        self.eval_every_steps = int(cfg.eval.eval_every_steps)
        self.num_eval_episodes = int(cfg.eval.num_eval_episodes)

    @staticmethod
    def _resolve_vec_env_cls(vec_env_name: str):
        if vec_env_name == "SubprocVecEnv":
            return SubprocVecEnv
        if vec_env_name == "DummyVecEnv":
            return DummyVecEnv
        raise ValueError("vec_env_cls must be one of {'SubprocVecEnv', 'DummyVecEnv'}")

    def _make_single_env(self, rank: int, task_name: str, render_mode=None):
        ''' 
        creates a single environment for task_name with unique
        seed and gymnasium compatibility wrapper
        '''
        seed = self.seed + rank
        task = task_name

        def _init():
            env = ContinualBenchEnv(render_mode=render_mode, seed=seed)
            env.set_task(task)
            wrapped_env = GymV21CompatibilityV0(env=env)
            wrapped_env.max_path_length = env.max_path_length
            return wrapped_env
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

    def _build_policy(self, env, num_envs: int): # building SAC and PPO policies soon.
        if self.cfg.policy == "RandomPolicy":
            return RandomPolicy(env.action_space, num_envs)
        if self.cfg.policy == "DDPG":
            return DDPG(self.cfg, env).reset()
        if self.cfg.policy == "SAC":
            return SAC(self.cfg, env).reset()
        # if self.cfg.policy == "PPO":
        #     return PPO(self.cfg, env).reset()
        raise ValueError(f"Unsupported policy={self.cfg.policy}")

    def train(self):
        # seeding
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.backends.cudnn.deterministic = self.cfg.torch_deterministic
        # device
        self.cfg.device = "cuda" if torch.cuda.is_available() and self.cfg.cuda else "cpu"
        print(f"Using device: {self.cfg.device}")
        # create the vectorized environments
        vec_envs = self.make_envs()
        training_order = list(vec_envs.keys())
        # to build the policy we need action space and observation space of the first task which is consistent across all tasks.
        first_env = vec_envs[training_order[0]]
        # building the policy will come from intitializing the agent first. i.e. calling SAC.reset()
        agent = self._build_policy(first_env, num_envs=self.num_envs)
        # does our policy need a replay buffer?
        replay_buffer_enabled = bool(self.cfg.get("replay_buffer", False))
        # random action sampling until the learning starts
        learning_starts = int(self.cfg.get("learning_starts", 0))
        if replay_buffer_enabled:
            first_env.observation_space.dtype = np.float32
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
        save_dir = Path("models") / run_name
        save_dir.mkdir(parents=True, exist_ok=True)
        logger = ContinualLogger(
            project=self.cfg.logging.project,
            run_name=run_name,
            enable_wandb=bool(self.cfg.logging.enable_wandb),
            entity=self.cfg.logging.get("wandb_entity", None),
            config=OmegaConf.to_container(self.cfg, resolve=True),
            num_envs=self.num_envs,
        )

        for task_idx, task_name in enumerate(training_order):
            vec_env = vec_envs[task_name]
            max_episode_steps = int(vec_env.get_attr("max_path_length")[0])
            # load weights from previous task before starting training on this one
            if task_idx > 0 and hasattr(agent, "load"):
                agent.load(str(save_dir))
                print(f"[Task {task_idx}] Loaded checkpoint from task {training_order[task_idx - 1]}")

            for ep in range(int(self.train_episodes_per_task)):
                obs = vec_env.reset() # reset the vec env
                episode_returns = np.zeros(vec_env.num_envs, dtype=np.float32)
                episode_lengths = np.zeros(vec_env.num_envs, dtype=np.int32)

                for t in range(max_episode_steps):
                    if logger.global_step < learning_starts:
                        action_space = vec_env.action_space
                        actions = np.array([action_space.sample() for _ in range(vec_env.num_envs)])
                    else:
                    # we can specify updates here and task specific buffer stuff. like if policy is sac warmup buffer and append to buffer
                    # HERE
                        actions = agent.predict(obs)
                    next_obs, rewards, dones, infos = vec_env.step(actions)
                    episode_returns += rewards
                    episode_lengths += 1
                    successes = np.array(
                        [float(info["success"]) for info in infos],
                        dtype=np.float32,
                    )
                    terminated = np.logical_and(dones, successes.astype(bool))
                    #SB3 terminal observation handling not relevant in continualbench
                    real_next_obs = next_obs.copy()
                    for idx, done in enumerate(dones):
                        if done:
                            real_next_obs[idx] = infos[idx]["terminal_observation"]
                    if replay_buffer_enabled:
                        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

                    obs = next_obs
                    # more logging of CRL metrics
                    logger.update_online(
                        task_name=task_name,
                        task_idx=task_idx,
                        episode_idx=ep,
                        timestep_in_episode=t,
                        successes=successes,
                    )

                    for idx, done in enumerate(dones):
                        if not done:
                            continue
                        logger.log_metrics(
                            {
                                "charts/episodic_return": float(episode_returns[idx]),
                                "charts/episodic_length": int(episode_lengths[idx]),
                            }
                        )
                        logger.on_episode_end(
                            task_name=task_name,
                            success=bool(infos[idx]["success"]),
                        )
                        episode_returns[idx] = 0.0
                        episode_lengths[idx] = 0

                    if (
                        replay_buffer_enabled
                        and logger.global_step > learning_starts
                        and hasattr(agent, "update")
                    ):
                        data = rb.sample(self.cfg.batch_size)
                        algorithm_metrics = agent.update(data, logger.global_step)
                        logger.log_algorithm_metrics(algorithm_metrics, step=logger.global_step)
            vec_env.close()
            # freeze this task's episodic success rate and update AP over completed tasks
            final_rate, ap_completed = logger.on_task_end(task_name)
            print(f"[Task {task_idx}] {task_name} final episodic success={final_rate:.3f}  AP over completed tasks={ap_completed:.3f}")
            # save checkpoint after finishing this task
            if hasattr(agent, "save"):
                agent.save(str(save_dir))
                print(f"[Task {task_idx}] Saved checkpoint after task {task_name}")

        logger.finish()


def main():
    cfg_dir = Path(__file__).resolve().parent.parent / "cfgs"
    cfg = parse_cfg(cfg_dir)
    print(OmegaConf.to_yaml(cfg))
    vecenv = ContinualBenchVecEnv(cfg)
    vecenv.train()


if __name__ == "__main__":
    main()