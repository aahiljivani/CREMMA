from random import seed
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from continual_bench.envs import ContinualBenchEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
from continual_bench.envs import ContinualBenchEnv

class ContinualBenchVecEnv:
    def __init__(self, num_envs, seed=0, parallel=True, task="Button"):
        self.task = task
        self.task_list = ["button", "door", "window", "faucet", "peg", "block"]
        self.num_envs = num_envs
        self.seed = seed
        self.parallel = parallel
        self.env = self._build_env()

    def make_env(self,env_id, seed=0, task="Button"):
        """
        Utility function for multiplexed env execution
        """
        def _init():
            env = ContinualBenchEnv(render_mode="rgb_array", seed=seed + env_id, set_task=task)
            # Explicitly wrap the env so its old-gym API is fully bridged 
            # to the new Gymnasium API expected by Stable Baselines 3
            return GymV21CompatibilityV0(env=env)
        return _init

    def _build_env(self):
        env_fns = [self.make_env(i, self.seed, self.task) for i in range(self.num_envs)]
        if self.parallel:
            env = SubprocVecEnv(env_fns)
        else:
            env = DummyVecEnv(env_fns)
        return env

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if action.shape[0] != self.num_envs:
            raise ValueError(f"Action shape must be {self.num_envs}, got {action.shape[0]}")
        return self.env.step(action)

    def observation_space(self):
        return self.env.observation_space

    def action_space(self):
        return self.env.action_space()

    def render(self):
        # Note: To render from SubprocVecEnv, you can use:
        # pixels = vec_env.get_images() after calling reset() and step()
        # The manual `mujoco.Renderer(env.model, ...)` is avoided here 
        # since the actual environment instances live in subprocesses.
        return self.env.get_images()

    def close(self):
        return self.env.close()



        
        