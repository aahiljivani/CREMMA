import random
from typing import Dict, List, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from continual_bench.envs import ContinualBenchEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

from logger import ContinualLogger


class RandomPolicy:
    def __init__(self, action_space, num_envs: int):
        self.action_space = action_space
        self.num_envs = num_envs

    def predict(self, obs, deterministic=True):
        return np.array([self.action_space.sample() for _ in range(self.num_envs)])


class ContinualBenchRunner:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.seed = cfg.seed
        self.num_envs = cfg.env.num_envs
        self.vec_env_cls = SubprocVecEnv if cfg.env.vec_env == "subproc" else DummyVecEnv
        self.task_sequence = list(cfg.env.task_sequence)
        self.benchmark_mode = cfg.benchmark.mode
        self.single_task_name = cfg.benchmark.single_task_name

    def _make_single_env(self, rank: int, task_name: str):
        def _init():
            env = ContinualBenchEnv(render_mode=None, seed=self.seed + rank)
            env.set_task(task_name)
            return GymV21CompatibilityV0(env=env)
        return _init

    def _build_training_order(self) -> List[str]:
        if self.benchmark_mode == "sequential":
            return list(self.task_sequence)
        if self.benchmark_mode == "random":
            tasks = list(self.task_sequence)
            rng = random.Random(self.seed)
            rng.shuffle(tasks)
            return tasks
        if self.benchmark_mode == "task":
            if self.single_task_name not in self.task_sequence:
                raise ValueError(f"single_task_name={self.single_task_name} must be one of {self.task_sequence}")
            return [self.single_task_name]
        raise ValueError(f"Unknown benchmark.mode: {self.benchmark_mode}")

    def make_envs(self) -> Dict[str, DummyVecEnv]:
        training_order = self._build_training_order()
        vec_envs = {}
        for task_name in training_order:
            env_fns = [self._make_single_env(i, task_name) for i in range(self.num_envs)]
            vec_envs[task_name] = self.vec_env_cls(env_fns)
        return vec_envs

    def evaluate_seen_tasks(self, policy, seen_tasks: List[str]) -> Dict[str, float]:
        task_scores = {}
        for task_name in seen_tasks:
            eval_env = DummyVecEnv([self._make_single_env(0, task_name)])
            eval_policy = RandomPolicy(eval_env.action_space, 1)
            episode_scores = []
            for _ in range(self.cfg.eval.num_eval_episodes):
                obs = eval_env.reset()
                done = [False]
                step_scores = []
                while not done[0]:
                    action = eval_policy.predict(obs, deterministic=True)
                    obs, rewards, done, infos = eval_env.step(action)
                    step_scores.append(float(infos[0][task_name]["success"]))
                episode_scores.append(float(np.mean(step_scores)) if step_scores else 0.0)
            eval_env.close()
            task_scores[task_name] = float(np.mean(episode_scores)) if episode_scores else 0.0
        return task_scores

    def train(self):
        vec_envs = self.make_envs()
        training_order = list(vec_envs.keys())
        first_env = vec_envs[training_order[0]]
        policy = RandomPolicy(first_env.action_space, self.num_envs)

        run_name = f"{self.cfg.algorithm.name}_{self.cfg.benchmark.mode}_{self.cfg.env.num_envs}env_seed{self.seed}"
        logger = ContinualLogger(
            project=self.cfg.logging.project,
            run_name=run_name,
            enable_wandb=self.cfg.logging.enable_wandb,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

        seen_tasks = []
        final_scores = {}

        for task_idx, task_name in enumerate(training_order):
            seen_tasks.append(task_name)
            vec_env = vec_envs[task_name]

            for ep in range(self.cfg.train.episodes_per_task):
                obs = vec_env.reset()
                for t in range(self.cfg.train.timesteps_per_episode):
                    actions = policy.predict(obs, deterministic=False)
                    obs, rewards, dones, infos = vec_env.step(actions)
                    successes = np.array([float(info[task_name]["success"]) for info in infos], dtype=np.float32)
                    logger.update_online(
                        task_name=task_name,
                        task_idx=task_idx,
                        episode_idx=ep,
                        timestep_in_episode=t,
                        successes=successes,
                    )

                    if logger.global_step % self.cfg.eval.eval_every_steps == 0:
                        task_scores = self.evaluate_seen_tasks(policy, seen_tasks)
                        logger.log_evaluation(seen_tasks, task_scores)

            vec_env.close()

        final_scores = self.evaluate_seen_tasks(policy, seen_tasks)
        logger.log_evaluation(seen_tasks, final_scores)
        logger.finish(final_scores)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    runner = ContinualBenchRunner(cfg)
    runner.train()


if __name__ == "__main__":
    main()
