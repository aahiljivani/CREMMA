from continual_bench.envs import ContinualBenchEnv

# Create the environment
env = ContinualBenchEnv(render_mode="rgb_array", seed=0)

# Set a task and reset
env.set_task("faucet")
obs = env.reset()

# Run an episode (max 500 steps by default)
for step in range(500):
    action = env.action_space.sample()  # 4D: [dx, dy, dz, gripper_effort]
    obs, reward, done, info = env.step(action)
    print(f"obs: {obs}, reward: {reward}, done: {done}, info: {info}")

    if done:  # Truncated at max_path_length
        obs = env.reset()
        break

env.close()