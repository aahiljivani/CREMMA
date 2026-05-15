import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from continual_bench.envs import ContinualBenchEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

def make_env(env_id, seed=0):
    """
    Utility function for multiplexed env execution
    """
    def _init():
        env = ContinualBenchEnv(render_mode="rgb_array", seed=seed + env_id)
        # Explicitly wrap the env so its old-gym API is fully bridged 
        # to the new Gymnasium API expected by Stable Baselines 3
        wrapped_env = GymV21CompatibilityV0(env=env)
        return wrapped_env
    return _init

if __name__ == "__main__":
    num_envs = 4
    
    # Create a list of environment callables
    env_fns = [make_env(i) for i in range(num_envs)]
    
    # Initialize the SubprocVecEnv
    # This runs the environments in parallel separate processes
    vec_env = SubprocVecEnv(env_fns)
    
    obs = vec_env.reset()
    
    for i in range(6000):
        # Sample actions for all environments
        actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
        
        # Step through the vectorized environment
        # Note: SubprocVecEnv handles resetting automatically when an episode finishes
        obs, rewards, dones, infos = vec_env.step(actions)
        
        # Note: To render from SubprocVecEnv, you can use:
        # pixels = vec_env.get_images()
        # The manual `mujoco.Renderer(env.model, ...)` is avoided here 
        # since the actual environment instances live in subprocesses.
        if i % 1000 == 0:
            print(f"Step {i}: done: {dones}, info: {infos}")
    
    vec_env.close()