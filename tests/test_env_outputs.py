"""
Quick diagnostic script: run a random policy for N steps under both
DummyVecEnv and SubprocVecEnv and log all raw outputs to a text file.

Usage:
    python -m tests.test_env_outputs
"""

import os
os.environ.setdefault("MUJOCO_GL", "glfw")

import pprint
import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from continual_bench.envs import ContinualBenchEnv
from shimmy.openai_gym_compatibility import GymV21CompatibilityV0

# ── Config ────────────────────────────────────────────────────────────────────
TASK_NAME   = "door"
SEED        = 0
NUM_ENVS    = 2
N_STEPS     = 10
OUTPUT_FILE = Path(__file__).parent / "env_output_diagnostic.txt"
# ─────────────────────────────────────────────────────────────────────────────


def make_single_env(rank: int, task_name: str, seed: int):
    def _init():
        env = ContinualBenchEnv(seed=seed + rank)
        env.set_task(task_name)
        wrapped = GymV21CompatibilityV0(env=env)
        wrapped.max_path_length = env.max_path_length
        return wrapped
    return _init


def run_and_collect(vec_env_cls, task_name: str, n_steps: int, label: str):
    """Roll out a random policy for n_steps and return a list of step records."""
    env_fns = [make_single_env(i, task_name, SEED) for i in range(NUM_ENVS)]

    if vec_env_cls is SubprocVecEnv:
        vec_env = vec_env_cls(env_fns, start_method="spawn")
    else:
        vec_env = vec_env_cls(env_fns)

    obs = vec_env.reset()
    records = []

    print(f"\n{'='*60}")
    print(f"  {label}  |  task={task_name}  |  num_envs={NUM_ENVS}")
    print(f"{'='*60}")

    for step in range(n_steps):
        actions = np.array([vec_env.action_space.sample() for _ in range(NUM_ENVS)])
        next_obs, rewards, dones, infos = vec_env.step(actions)

        record = {
            "step":    step,
            "obs":     next_obs,
            "actions": actions,
            "rewards": rewards,
            "dones":   dones,
            "infos":   infos,
            # derived: does top-level 'success' key exist?
            "info_keys_env0":        list(infos[0].keys()),
            "top_level_success":     [info.get("success", "KEY_MISSING") for info in infos],
            "nested_task_success":   [info.get(task_name, {}).get("success", "KEY_MISSING") for info in infos],
            "TimeLimit.truncated":   [info.get("TimeLimit.truncated", "KEY_MISSING") for info in infos],
        }
        records.append(record)

        print(f"\n--- step {step} ---")
        print(f"  obs shape      : {next_obs.shape}")
        print(f"  actions shape  : {actions.shape}")
        print(f"  rewards        : {rewards}")
        print(f"  dones          : {dones}")
        print(f"  info keys[0]   : {record['info_keys_env0']}")
        print(f"  top-level success (infos[i]['success'])      : {record['top_level_success']}")
        print(f"  nested success  (infos[i][task]['success'])  : {record['nested_task_success']}")
        print(f"  TimeLimit.truncated                          : {record['TimeLimit.truncated']}")
        print(f"  full infos[0]  :")
        pprint.pprint(infos[0], indent=4)

        obs = next_obs

    vec_env.close()
    return records


def write_log(all_results: dict, output_path: Path):
    with open(output_path, "w") as f:
        f.write(f"ContinualBench env diagnostic\n")
        f.write(f"task={TASK_NAME}  num_envs={NUM_ENVS}  n_steps={N_STEPS}\n")
        f.write("=" * 80 + "\n\n")

        for label, records in all_results.items():
            f.write(f"\n{'='*60}\n")
            f.write(f"  {label}\n")
            f.write(f"{'='*60}\n\n")

            for rec in records:
                f.write(f"--- step {rec['step']} ---\n")
                f.write(f"  obs shape            : {rec['obs'].shape}\n")
                f.write(f"  actions shape        : {rec['actions'].shape}\n")
                f.write(f"  rewards              : {rec['rewards']}\n")
                f.write(f"  dones                : {rec['dones']}\n")
                f.write(f"  info keys [env 0]    : {rec['info_keys_env0']}\n")
                f.write(f"  top-level success    : {rec['top_level_success']}\n")
                f.write(f"  nested task success  : {rec['nested_task_success']}\n")
                f.write(f"  TimeLimit.truncated  : {rec['TimeLimit.truncated']}\n")
                f.write(f"  full infos[0]:\n")
                f.write(pprint.pformat(rec["infos"][0], indent=4) + "\n")
                f.write("\n")

    print(f"\nLog written to: {output_path}")


def main():
    all_results = {}

    for label, cls in [("DummyVecEnv", DummyVecEnv), ("SubprocVecEnv", SubprocVecEnv)]:
        try:
            records = run_and_collect(cls, TASK_NAME, N_STEPS, label)
            all_results[label] = records
        except Exception as e:
            print(f"\n[ERROR] {label} failed: {e}", file=sys.stderr)
            all_results[label] = [{"error": str(e)}]

    write_log(all_results, OUTPUT_FILE)


if __name__ == "__main__":
    main()
