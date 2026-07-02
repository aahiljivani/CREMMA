import os

os.environ.setdefault("MUJOCO_GL", "egl")

import random
from pathlib import Path

import numpy as np
import optuna
import torch
from omegaconf import OmegaConf

from src.buffer import ReplayBuffer
from src.train import ContinualBenchVecEnv

# ---------------------------------------------------------------------------
# Constants — adjust these to trade tuning speed vs. quality
# ---------------------------------------------------------------------------
TASK = "block"          # pick-cube analog in this codebase
N_TUNE_EPISODES = 400    # episodes per trial (vs. 100 in full training)
N_EVAL_EPISODES = 10    # greedy eval episodes used to score each trial
N_TUNE_ENVS = 10         # parallel envs per trial (fewer than default 10)
N_TRIALS = 50           # total Optuna trials


def build_cfg(params: dict, n_episodes: int = N_TUNE_EPISODES) -> OmegaConf:
    """Build an OmegaConf config for a single HPO trial without touching sys.argv."""
    cfg_dir = Path(__file__).resolve().parent.parent / "cfgs"
    base = OmegaConf.load(cfg_dir / "default.yaml")
    base.merge_with(OmegaConf.load(cfg_dir / "algorithms" / "sac.yaml"))
    base.merge_with(OmegaConf.load(cfg_dir / "task.yaml"))

    overrides = OmegaConf.create(
        {
            "single_task_name": TASK,
            "benchmark_mode": "task",
            "num_envs": N_TUNE_ENVS,
            "train": {"episodes_per_task": n_episodes},
            "logging": {"enable_wandb": False},
            **params,
        }
    )
    base.merge_with(overrides)
    return base


def train_for_hpo(cfg) -> float:
    """Run a condensed training pass on `block` and return its success rate."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = bool(cfg.torch_deterministic)

    cfg.device = "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"

    bench = ContinualBenchVecEnv(cfg)
    vec_envs = bench.make_envs()           # {TASK: VecEnv}
    vec_env = vec_envs[TASK]

    agent = bench._build_policy(vec_env, num_envs=bench.num_envs)
    rb = ReplayBuffer(cfg=cfg, env=vec_env)

    learning_starts = int(cfg.get("learning_starts", 0))
    max_episode_steps = int(vec_env.get_attr("max_path_length")[0])

    global_step = 0
    for _ in range(bench.train_episodes_per_task):
        obs = vec_env.reset()
        obs = bench._augment_obs(TASK, obs)
        for _ in range(max_episode_steps):
            if global_step < learning_starts:
                actions = np.array(
                    [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
                )
            else:
                actions = agent.predict(obs)

            next_obs, rewards, dones, infos = vec_env.step(actions)
            next_obs = bench._augment_obs(TASK, next_obs)
            global_step += vec_env.num_envs

            successes = np.array(
                [float(info["success"]) for info in infos], dtype=np.float32
            )
            terminated = np.logical_and(dones, successes.astype(bool))

            real_next_obs = next_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    real_next_obs[idx] = bench._augment_terminal_obs(
                        TASK, infos[idx]["terminal_observation"]
                    )

            rb.add(obs, actions, rewards, terminated, real_next_obs)
            obs = next_obs

            if global_step > learning_starts and hasattr(agent, "update"):
                for _ in range(vec_env.num_envs):
                    data = rb.sample(cfg.batch_size)
                    agent.update(data)

    vec_env.close()
    return bench.evaluate_task(TASK, agent, N_EVAL_EPISODES)


def sample_sac_params(trial: optuna.Trial) -> dict:
    return {
        "policy_lr": trial.suggest_float("policy_lr", 1e-5, 1e-3, log=True),
        "q_lr": trial.suggest_float("q_lr", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999, step=0.001),
        "tau": trial.suggest_float("tau", 0.001, 0.01, step=0.001),
        "batch_size": trial.suggest_int("batch_size", 32, 256, step=32),
        "learning_starts": trial.suggest_int("learning_starts", 1000, 10000, step=1000),
        "policy_frequency": trial.suggest_int("policy_frequency", 1, 4),
        "target_network_frequency": trial.suggest_int("target_network_frequency", 1, 4),
        "alpha": trial.suggest_float("alpha", 0.01, 1.0, step=0.01),
        "autotune": trial.suggest_categorical("autotune", [True, False]),
    }


def objective(trial: optuna.Trial) -> float:
    params = sample_sac_params(trial)
    cfg = build_cfg(params)
    return train_for_hpo(cfg)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="sac_block_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    print("\nBest trial:")
    best = study.best_trial
    print(f"  Success rate: {best.value:.3f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
