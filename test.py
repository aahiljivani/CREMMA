import mujoco
from continual_bench.envs import ContinualBenchEnv

env = ContinualBenchEnv(render_mode="rgb_array", seed=0)


renderer = mujoco.Renderer(env.model, 480, 480)

for i in range(2000):
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    renderer.update_scene(env.data)
    pixels = renderer.render()
    print(f"Step {i}: reward: {reward}")
    
    if done:
        env.reset()