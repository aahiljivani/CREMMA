import random
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from continual_bench.envs import ContinualBenchEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0


class ContinualBenchVecEnv:
    def __init__(self, num_envs, seed=0, vec_env_cls=SubprocVecEnv, task="sequential"):
        self.num_envs = num_envs
        self.seed = seed
        self.vec_env_cls = vec_env_cls
        self.task = task

        self.task_sequence = ["button", "door", "window", "faucet", "peg", "block"]
        self.task_list = self.task_sequence + ["sequential", "random"]

    def _make_single_env(self, rank, task_name):
        def _init():
            env = ContinualBenchEnv(render_mode=None, seed=self.seed + rank)
            env.set_task(task_name)
            return GymV21CompatibilityV0(env=env)
        return _init

    def make_envs(self):
        vec_envs = {}

        if self.task in self.task_sequence:
            env_fns = [self._make_single_env(i, self.task) for i in range(self.num_envs)]
            vec_envs[self.task] = self.vec_env_cls(env_fns)

        elif self.task == "sequential":
            for task_name in self.task_sequence:
                env_fns = [self._make_single_env(i, task_name) for i in range(self.num_envs)]
                vec_envs[task_name] = self.vec_env_cls(env_fns)

        elif self.task == "random":
            tasks = self.task_sequence[:]
            random.shuffle(tasks)
            for task_name in tasks:
                env_fns = [self._make_single_env(i, task_name) for i in range(self.num_envs)]
                vec_envs[task_name] = self.vec_env_cls(env_fns)

        else:
            raise ValueError(f"Task {self.task} not found. Available tasks: {self.task_list}")

        return vec_envs

    def train(self, episodes=1000):
        vec_envs = self.make_envs()

        if self.task in self.task_sequence:
            vec_env = vec_envs[self.task]
            for ep in range(episodes):
                obs = vec_env.reset()
                for step in range(500):
                    actions = np.array([vec_env.action_space.sample() for _ in range(self.num_envs)])
                    obs, rewards, dones, infos = vec_env.step(actions)
                    print(f"rewards: {rewards}")
                vec_env.close()

        elif self.task in ["sequential", "random"]:
            for task_name, vec_env in vec_envs.items():
                print(f"Training on task: {task_name}")
                for ep in range(episodes):
                    obs = vec_env.reset()
                    for step in range(500):
                        actions = np.array([vec_env.action_space.sample() for _ in range(self.num_envs)])
                        obs, rewards, dones, infos = vec_env.step(actions)
                        print(f"infos: {infos}")
                vec_env.close()                
if __name__ == "__main__":
    vec_env = ContinualBenchVecEnv(num_envs=10, task="sequential")
    vec_env.train()
