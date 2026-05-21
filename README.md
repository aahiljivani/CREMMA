<div align="center">

# Continual RL Evaluation of Model Based and Model Free Algorithms (CREMMA)

**This repository extends the public ContinualBench benchmark from sail-sg for additional sequential, and task benchmarking with parallel and vectorized environment training infrastructure, training workflows, and comparative evaluation of continual RL algorithms. The repo consists of benchmark algorithms:
# Model-Free Continual RL Agents:**
- PackNet
- CompoNet
# Model-Based RL Continual RL Agents
- DRAGO
- HyperCRL

**Original benchmark:**
- Paper: Continual Reinforcement Learning by Planning with Online World Models
- Repo: https://github.com/sail-sg/ContinualBench

**A Unified Benchmark for Continual Reinforcement Learning with Shared World Dynamics**

</div>

<p align="center">
  <img src="./illustration.jpg" width=80%/>
</p>

Continual Bench provides a single MuJoCo environment containing six distinct manipulation tasks for a Sawyer robot arm. All tasks coexist in the same physical scene with shared dynamics, making it ideal for evaluating online reinforcement learning agents under a continual learning setup — where the agent must sequentially learn new tasks without forgetting old ones.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
  - [Available Tasks](#available-tasks)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward Structure](#reward-structure)
- [Customization](#customization)
  - [Setting a Specific Task](#setting-a-specific-task)
  - [Episode Length (Max Steps)](#episode-length-max-steps)
  - [Render Modes](#render-modes)
  - [Seeding for Reproducibility](#seeding-for-reproducibility)
  - [Task Sequences (Continual Learning)](#task-sequences-continual-learning)
- [Vectorized Environments](#vectorized-environments)
  - [Using ContinualBenchVecEnv (Recommended)](#using-continualbenchvecenv-recommended)
  - [Manual SubprocVecEnv Setup](#manual-subprocvecenv-setup)
- [Keyboard Control (Interactive Testing)](#keyboard-control-interactive-testing)
- [API Reference](#api-reference)
- [License](#license)
- [Citation](#citation)

---

## Installation

```bash
git clone https://github.com/aahiljivani/CREMMA.git && cd CREMMA
pip install -e .
```

**Dependencies:** MuJoCo (`mujoco`), NumPy, PyTorch, OpenAI Gym (`gym`), and `glfw==2.5.0` (installed automatically).

For vectorized training with Stable Baselines 3, also install:

```bash
pip install stable-baselines3 shimmy
```

---

## Quick Start

The recommended way to run the full benchmark suite with environment vectorization, config parsing, and wandb logging is via the provided training script:

```bash
# Run the continual learning benchmark using the default configuration
python -m src.train
```

For manual, lower-level control over the environment:
```python
from continual_bench.envs import ContinualBenchEnv

# Create the environment
env = ContinualBenchEnv(render_mode="rgb_array", seed=0)

# Set a task and reset
env.set_task("button")
obs = env.reset()

# Run an episode (max 500 steps by default)
for step in range(500):
    action = env.action_space.sample()  # 4D: [dx, dy, dz, gripper_effort]
    obs, reward, done, info = env.step(action)

    if done:  # Truncated at max_path_length
        obs = env.reset()
        break

env.close()
```

> **Note:** The `step()` method returns a **4-tuple** `(obs, reward, done, info)` following the legacy OpenAI Gym API. The `reward` is a dictionary keyed by task name, and `info` contains per-task success metrics. The episode is **never terminated early** — `done` is only `True` when the step count reaches `max_path_length` (truncation).

---

## Environment Details

### Available Tasks

All six tasks share a single Sawyer robot arm and a common physical scene:

| Index | Task Name | Description | Success Criterion |
|:-----:|:---------:|:------------|:-----------------:|
| 0 | `button` | Press a button downward | `obj_to_target ≤ 0.024` |
| 1 | `door` | Open a door by its handle | `obj_to_target ≤ 0.08` |
| 2 | `window` | Slide a window open | `target_to_obj ≤ 0.05` |
| 3 | `faucet` | Turn a faucet handle | `target_to_obj ≤ 0.07` |
| 4 | `peg` | Insert a peg into a hole | `obj_to_target ≤ 0.07` |
| 5 | `block` | Push/pick a block to a target | `obj_to_midpoint ≤ 0.05` |

Access the full list programmatically:

```python
print(env.all_tasks)  # ['button', 'door', 'window', 'faucet', 'peg', 'block']
```

### Observation Space

The observation is a **26-dimensional** `float64` vector containing:

| Indices | Description |
|:-------:|:------------|
| `0:3` | End-effector (hand) position `[x, y, z]` |
| `3` | Gripper openness (normalized `0.0`–`1.0`) |
| `4:7` | Button position |
| `7:10` | Door handle position |
| `10` | Door joint angle |
| `11:14` | Window handle position |
| `14:17` | Faucet handle position |
| `17:20` | Peg end position |
| `20:23` | Block position |
| `23` | Gripper direction (delta openness) |
| `24` | Left pad Y offset from hand |
| `25` | Right pad Y offset from hand |

> All object states are present in every observation regardless of the active task, enabling the agent to observe all tasks simultaneously.

### Action Space

A **4-dimensional** continuous `Box([-1, -1, -1, -1], [+1, +1, +1, +1])`:

| Index | Description |
|:-----:|:------------|
| `0` | Δx (end-effector position delta) |
| `1` | Δy (end-effector position delta) |
| `2` | Δz (end-effector position delta) |
| `3` | Gripper effort (`-1` = open, `+1` = close) |

Actions are scaled by `action_scale = 1/100` before being applied to the mocap body.

### Reward Structure

`env.step()` returns rewards as a **dictionary** keyed by task name. Each task's reward is in the range `[0, 10]` (with the exception of `block` which can return `-10` if the block falls off the table). Each task uses a shaped reward composed of reach, grasp, and in-place components.

```python
obs, reward_dict, done, info = env.step(action)

# Access reward for the active task
current_reward = reward_dict["door"]

# Access success metrics
door_info = info["door"]
print(door_info["success"])       # bool: whether task is solved
print(door_info["obj_to_target"]) # float: distance to goal
```

---

## Customization

### Setting a Specific Task

Use `set_task()` to select which task the robot should focus on. This determines the hand's initial position and which camera angle is active. **You must call `reset()` after switching tasks.**

```python
env = ContinualBenchEnv(render_mode="rgb_array", seed=0)

# Switch to a specific task
env.set_task("window")  # One of: 'button', 'door', 'window', 'faucet', 'peg', 'block'
env.reset()
```

The task can also be set at construction time when using the vectorized wrapper:

```python
from vectorize import ContinualBenchVecEnv

# All parallel environments will run the "door" task
vec_env = ContinualBenchVecEnv(num_envs=4, task="door")
```

> **Default behavior:** If `set_task()` is never called explicitly, the environment initializes to the first task (`"button"`) by default.

### Episode Length (Max Steps)

Each episode is **fixed at 500 steps** (`max_path_length = 500`). The episode is truncated (not terminated) when this limit is reached. This matches the experimental protocol from the paper.

```python
# The max step count is a class-level attribute
print(env.max_path_length)  # 500

# To use a custom episode length, override the attribute after construction:
env = ContinualBenchEnv(render_mode="rgb_array", seed=0)
env.max_path_length = 200  # Shorter episodes for faster iteration
```

> **Important:** The environment raises a `ValueError` if you call `step()` after `max_path_length` is reached without calling `reset()` first. When `done=True` is returned, you must reset manually.

### Render Modes

```python
# Off-screen rendering (for training / recording)
env = ContinualBenchEnv(render_mode="rgb_array", seed=0)
pixels = env.render()  # Returns an HxWx3 numpy array

# On-screen rendering (for debugging / visualization)
env = ContinualBenchEnv(render_mode="human", seed=0)

# Depth buffer rendering
env = ContinualBenchEnv(render_mode="depth_array", seed=0)
```

The render resolution defaults to **480×480** pixels. The default render FPS is **80**.

### Seeding for Reproducibility

```python
# Seed controls the random state for action/observation/goal spaces
env = ContinualBenchEnv(render_mode="rgb_array", seed=42)
```

When `seed` is provided, the global `np.random` state is temporarily modified during initialization and then restored, so the seed does not leak into other parts of your code.

### Task Sequences (Continual Learning)

For continual learning experiments, switch tasks sequentially upon success:

```python
env = ContinualBenchEnv(render_mode="rgb_array", seed=0)

task_sequence = ["door", "window", "button", "faucet", "peg", "block"]
task_idx = 0

env.set_task(task_sequence[task_idx])
obs = env.reset()

for step in range(10000):
    action = policy(obs)  # Your RL agent
    obs, reward_dict, done, info = env.step(action)

    current_task = task_sequence[task_idx]

    # Check if the current task was solved
    if info[current_task]["success"]:
        task_idx += 1
        if task_idx >= len(task_sequence):
            print("All tasks completed!")
            break
        env.set_task(task_sequence[task_idx])
        obs = env.reset()

    elif done:  # Truncated at 500 steps
        obs = env.reset()
```

---

## Vectorized Environments & Benchmarking

### Using the ContinualBenchRunner (Recommended)

The `ContinualBenchRunner` (invoked via `src/train.py`) is the main entry point for running parallel vectorized environments and evaluating continual learning agents. It leverages an `OmegaConf`-based configuration system.

The default settings are located in `cfgs/default.yaml`:

```yaml
benchmark_mode: continual      # Modes: "continual" (sequential), "random" (shuffled), or "task" (single task)
task_list: ["door", "window", "button", "faucet", "peg", "block"]
seed: 0
num_envs: 10
vec_env_cls: SubprocVecEnv     # Use DummyVecEnv for single-process debugging
policy: RandomPolicy

logging:
  project: continual-bench
  enable_wandb: true           # Set to true to automatically sync metrics
  wandb_entity: jivani-aahil-university-of-ottawa

train:
  episodes_per_task: 100
  timesteps_per_episode: 500

eval:
  eval_every_steps: 1000
  num_eval_episodes: 10
```

You can launch a fully vectorized training session by running:
```bash
python -m src.train
```

The runner automatically configures the `SubprocVecEnv` workers, applies the correct base `seed`, bridges the OpenAI Gym API to Gymnasium via `shimmy`, and logs step successes, regret, and evaluation metrics directly to Weights & Biases (`wandb`).

### Manual SubprocVecEnv Setup

For full control over the vectorized setup:

```python
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from continual_bench.envs import ContinualBenchEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

def make_env(env_id, seed=0):
    def _init():
        env = ContinualBenchEnv(render_mode="rgb_array", seed=seed + env_id)
        env.set_task("door")  # Set your desired task
        return GymV21CompatibilityV0(env=env)
    return _init

num_envs = 4
vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
obs = vec_env.reset()
# ... training loop ...
vec_env.close()
```

---

## Keyboard Control (Interactive Testing)

An interactive keyboard controller is included for debugging and task visualization:

```bash
python tests/keyboard_control.py
```

**Key Bindings:**

| Key | Action |
|:---:|:-------|
| `W/A/S/D` | Move end-effector in XY plane |
| `K/J` | Move end-effector up/down (Z axis) |
| `H` | Close gripper |
| `L` | Open gripper |
| `R` | Reset environment |
| `0–5` | Switch task (0=button, 1=door, ..., 5=block) |

---

## API Reference

### `ContinualBenchEnv(render_mode=None, seed=None)`

| Method | Description |
|:------:|:------------|
| `set_task(task_name: str)` | Set active task. One of `'button'`, `'door'`, `'window'`, `'faucet'`, `'peg'`, `'block'`. Must call `reset()` after. |
| `reset()` | Reset environment to initial state for current task. Returns observation. |
| `step(action)` | Execute action. Returns `(obs, reward_dict, done, info)`. |
| `render()` | Render current frame according to `render_mode`. |
| `get_init_state()` | Get initial state data (useful for model-based RL). |
| `close()` | Clean up renderer resources. |

| Attribute | Type | Description |
|:---------:|:----:|:------------|
| `all_tasks` | `list[str]` | All available task names |
| `task_spec` | `TaskSpec` | Current task's name, hand init position, and object init position |
| `max_path_length` | `int` | Maximum steps per episode (default: `500`) |
| `action_space` | `Box` | 4D continuous action space |
| `observation_space` | `Box` | Observation space |

---

## License

`continual-bench` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Citation

If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{liu2025continual,
  title={Continual Reinforcement Learning by Planning with Online World Models},
  author={Liu, Zichen and Fu, Guoji and Du, Chao and Lee, Wee Sun and Lin, Min},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
