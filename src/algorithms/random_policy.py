import numpy as np
class RandomPolicy:
    def __init__(self, action_space, num_envs: int):
        self.action_space = action_space
        self.num_envs = num_envs

    def predict(self, obs, deterministic=True):
        return np.array([self.action_space.sample() for _ in range(self.num_envs)])

    def save(self):
        pass

    def load(self):
        pass
