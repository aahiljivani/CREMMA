import os

# Select an offscreen OpenGL backend BEFORE any mujoco import so eval rendering
# works on headless servers (no DISPLAY). EGL uses the GPU; set MUJOCO_GL=osmesa
# for a CPU fallback if EGL is unavailable. setdefault respects a user override.
os.environ.setdefault("MUJOCO_GL", "egl")

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from gym.spaces import Box
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from continual_bench.envs import ContinualBenchEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
import torch
import torch.optim as optim
from .cfg import parse_cfg
from .logger import ContinualLogger
from src.algorithms import DDPG, RandomPolicy, SAC, RND_SAC
from src.buffer import ReplayBuffer, ExpertBuffer




class ContinualBenchVecEnv:
    GCRL_GOAL_DIM = 4

    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = int(cfg.seed)
        self.num_envs = int(cfg.num_envs)
        self.task_list = list(cfg.task_list)
        self.benchmark_mode = str(cfg.benchmark_mode)
        self.single_task_name = cfg.get("single_task_name", None)
        self.vec_env_cls = self._resolve_vec_env_cls(cfg.vec_env_cls)
        self.train_episodes_per_task = int(cfg.train.episodes_per_task)
        self.eval_video_steps = int(cfg.eval.eval_video_steps)
        self.num_eval_episodes = int(cfg.eval.num_eval_episodes)
        self.replay_buffer_enabled = cfg.replay_buffer_enabled
        self.expert_buffer_enabled = cfg.expert_buffer_enabled
        self.gcrl = bool(cfg.get("gcrl", False))
        self.target_pos = dict()

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

    def _compute_goal_vector(self, task_name: str) -> np.ndarray:
        env = ContinualBenchEnv(seed=self.seed)
        env.set_task(task_name)
        env.reset()
        # get the target position for the task as opposed to one hot encoded task id
        target_pos = env.init_data[env.task_spec.name].target_pos.astype(np.float32)
        # if task is door then we need to add the door angle to the goal vector
        door_angle = np.pi / 2 + np.pi / 6 if task_name == "door" else 0.0
        goal = np.concatenate(
            [target_pos, np.array([door_angle], dtype=np.float32)]
        ).astype(np.float32)
        env.close()
        return goal

    def _ensure_goal_vector(self, task_name: str) -> np.ndarray:
        '''
        return dictionary of task name to goal vector. This is used to augment the observation space
        with a task id that is informative for the policy instead of one hot encoded task id.
        '''
        if task_name not in self.target_pos:
            self.target_pos[task_name] = self._compute_goal_vector(task_name)
        return self.target_pos[task_name]

    def _augment_obs(self, task_name: str, obs: np.ndarray) -> np.ndarray:
        if not self.gcrl:
            return obs
        goal = self._ensure_goal_vector(task_name)
        goals = np.repeat(goal[None, :], obs.shape[0], axis=0)
        return np.concatenate([obs, goals], axis=1).astype(np.float32)

    def _augment_terminal_obs(self, task_name: str, terminal_obs: np.ndarray) -> np.ndarray:
        if not self.gcrl:
            return terminal_obs
        goal = self._ensure_goal_vector(task_name)
        return np.concatenate([terminal_obs, goal]).astype(np.float32)

    def _expand_observation_space(self, vec_env):
        ''' 
        so the environment obs space is accurate for the policy when it takes in env.observation_space.
        '''
        if not self.gcrl:
            return vec_env
        obs_space = vec_env.observation_space
        low = np.concatenate(
            [obs_space.low, np.full(self.GCRL_GOAL_DIM, -np.inf, dtype=np.float32)]
        )
        high = np.concatenate(
            [obs_space.high, np.full(self.GCRL_GOAL_DIM, np.inf, dtype=np.float32)]
        )
        vec_env.observation_space = Box(low=low, high=high, dtype=np.float32)
        return vec_env

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
            if self.gcrl:
                self._ensure_goal_vector(task_name)
            env_fns = [self._make_single_env(i, task_name) for i in range(self.num_envs)]
            vec_envs[task_name] = self.vec_env_cls(env_fns, start_method="spawn")
            self._expand_observation_space(vec_envs[task_name])
        return vec_envs

    def _build_policy(self, env, num_envs: int): # building SAC and PPO policies soon.
        if self.cfg.policy == "RandomPolicy":
            return RandomPolicy(env.action_space, num_envs)
        elif self.cfg.policy == "DDPG":
            return DDPG(self.cfg, env).reset()
        elif self.cfg.policy == "SAC":
            return SAC(self.cfg, env).reset()
        elif self.cfg.policy == "RND_SAC":
            return RND_SAC(self.cfg, env).reset()
        else:
            raise ValueError(f"Unsupported policy={self.cfg.policy}")

    def record_video(self, task_name, agent, logger):
        '''
        record a video of the policy performing the task
        '''
        eval_env = DummyVecEnv([self._make_single_env(0, task_name, render_mode="rgb_array")])
        self._expand_observation_space(eval_env)
        obs = eval_env.reset()
        obs = self._augment_obs(task_name, obs)
        done = [False]
        frames = []
        while not done[0]:
            actions = agent.predict(obs)
            obs, _, done, _ = eval_env.step(actions)
            obs = self._augment_obs(task_name, obs)
            frames.append(eval_env.envs[0].gym_env.render())
        eval_env.close()
        if frames:
            logger.log_video(task_name, frames)

    def evaluate_task(self, task_name, agent, n_episodes):
        # rank offset keeps eval seeds disjoint from the training env seeds
        eval_env = DummyVecEnv([self._make_single_env(10_000, task_name)])
        self._expand_observation_space(eval_env)
        successes = 0
        for _ in range(n_episodes):
            obs = eval_env.reset()
            obs = self._augment_obs(task_name, obs)
            done = [False]
            ep_success = False
            while not done[0]:
                actions = agent.predict(obs)
                obs, _, done, infos = eval_env.step(actions)
                obs = self._augment_obs(task_name, obs)
                if bool(infos[0].get("success", 0.0)):
                    ep_success = True
            successes += int(ep_success)
        eval_env.close()
        return successes / max(n_episodes, 1)

    def evaluate_seen_tasks(self, seen_tasks, agent):
        # current shared policy evaluated on every task seen so far -> p_tau(w)
        return {t: self.evaluate_task(t, agent, self.num_eval_episodes) for t in seen_tasks}

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
        # maybe we need a buffer type?
        # random action sampling until the learning starts
        learning_starts = int(self.cfg.get("learning_starts", 0))
        if self.replay_buffer_enabled:
            rb = ReplayBuffer(cfg=self.cfg, env=first_env)

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
        seen_tasks = []
        for task_idx, task_name in enumerate(training_order):
            vec_env = vec_envs[task_name]
            max_episode_steps = int(vec_env.get_attr("max_path_length")[0])
            if self.replay_buffer_enabled:
                rb.reset()
            # load weights from previous task before starting training on this one
            if task_idx > 0 and hasattr(agent, "load"):
                agent.load(str(save_dir))
                print(f"[Task {task_idx}] Loaded checkpoint from task {training_order[task_idx - 1]}")

            for ep in range(int(self.train_episodes_per_task)):
                obs = vec_env.reset() # reset the vec env
                obs = self._augment_obs(task_name, obs)
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
                    next_obs = self._augment_obs(task_name, next_obs)
                    episode_returns += rewards
                    episode_lengths += 1
                    successes = np.array(
                        [float(info["success"]) for info in infos],
                        dtype=np.float32,
                    )
                    # if both done and success then the episode is terminated or if just done is true as well
                    terminated = np.logical_and(dones, successes.astype(bool))
                    
                    real_next_obs = next_obs.copy()
                    for idx, done in enumerate(dones):
                        if done:
                            real_next_obs[idx] = self._augment_terminal_obs(
                                task_name, infos[idx]["terminal_observation"]
                            )
                    if self.replay_buffer_enabled:
                        rb.add(obs, actions, rewards, terminated, real_next_obs)

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
                            success=bool(infos[idx].get("success", False)),
                            episode_length=int(episode_lengths[idx]),
                        )
                        episode_returns[idx] = 0.0
                        episode_lengths[idx] = 0

                    if (
                        self.replay_buffer_enabled
                        and logger.global_step > learning_starts
                        and hasattr(agent, "update")
                    ):
                        for _ in range(self.num_envs):
                            data = rb.sample(self.cfg.batch_size)
                            algorithm_metrics = agent.update(data)
                        logger.log_algorithm_metrics(algorithm_metrics, step=logger.global_step)

                    if logger.run is not None and logger.global_step % int(self.cfg.eval.eval_video_steps) == 0:
                        self.record_video(task_name, agent, logger)
            vec_env.close()
            logger.on_task_end(task_name)
            seen_tasks.append(task_name)
            # # we add rnd_sac here
            # if self.cfg.rnd and self.cfg.policy == "SAC":
            #     rnd_sac = RND_SAC(self.cfg, vec_env, task_name)
            #     rnd_sac.train()
            # elif self.cfg.rnd and self.cfg.policy == "PPO":
            #     rnd_ppo = RND_PPO(self.cfg, vec_env, task_name)
            #     rnd_ppo.train()

            per_task = self.evaluate_seen_tasks(seen_tasks, agent)
            ap = logger.log_offline_ap(per_task)
            print(f"[Task {task_idx}] {task_name} done. AP(w)={ap:.3f}  per-task={per_task}")
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