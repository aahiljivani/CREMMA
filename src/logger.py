import numpy as np
import wandb
import time


class ContinualLogger:
    def __init__(self, project, config, run_name=None, enable_wandb=True, entity=None, num_envs=1):
        self.enable_wandb = enable_wandb
        self.num_envs = int(num_envs)
        self.global_step = 0
        self.regret_sum = 0.0
        self.regret_steps = 0
        self._last_ap = None
        self.run = None
        self.start_time = time.time()
        # per-task episodic success tracking (reset each task)
        self._task_episodes = 0
        self._task_successes = 0
        # frozen diagonal p_tau(tau): learning perf, NOT AP
        self._completed_task_diag: dict = {}
        if self.enable_wandb:
            self.run = wandb.init(project=project, entity=entity, name=run_name, config=config)
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step")
            self.run.define_metric("regret_running", summary="last")
            self.run.define_metric("AP", summary="last")

    @staticmethod
    def step_success(successes):
        return float(np.mean(np.asarray(successes, dtype=np.float32)))

    def update_online(self, task_name, task_idx, episode_idx, timestep_in_episode, successes):
        online_success = self.step_success(successes)
        self.global_step += self.num_envs
        payload = {
            "global_step": self.global_step,
            "task_idx": int(task_idx),
            "episode_idx": int(episode_idx),
            "timestep_in_episode": int(timestep_in_episode),
            f"train_step_success_{task_name}": online_success,
        }
        if self.run is not None:
            self.run.log(payload)
        return online_success

    def on_episode_end(self, task_name: str, success: bool, episode_length: int):
        self._task_episodes += 1
        if success:
            self._task_successes += 1
        rate = self._task_successes / self._task_episodes

        # Regret, Eq. 9: time-average of (1 - online success) over the whole run.
        # Per-episode 0/1 outcome, weighted by episode length so the running mean
        # approximates the integral over env-steps. Accumulates across all tasks.
        self.regret_sum += (1.0 - float(success)) * int(episode_length)
        self.regret_steps += int(episode_length)
        regret_running = self.regret_sum / self.regret_steps

        payload = {
            "global_step": self.global_step,
            "regret_running": float(regret_running),
            "train_task_episodic_success": float(rate),
            f"train_episodic_success_{task_name}": float(rate),
        }
        if self.run is not None:
            self.run.log(payload)
        return float(rate), float(regret_running)

    def on_task_end(self, task_name: str):
        final_rate = self._task_successes / self._task_episodes if self._task_episodes > 0 else 0.0
        self._completed_task_diag[task_name] = final_rate  # p_tau(tau), learning perf
        payload = {
            "global_step": self.global_step,
            f"train_final_success_{task_name}": final_rate,
            "num_completed_tasks": len(self._completed_task_diag),
        }
        if self.run is not None:
            self.run.log(payload)
        self._task_episodes = 0
        self._task_successes = 0
        return final_rate
    
    def log_offline_ap(self, per_task_success: dict):
        """
        Average Performance AP(w), Eq. 8: mean over all seen tasks of the CURRENT
        policy's offline success rate on each task. `per_task_success` is
        {task_name: success_rate} from evaluating the current agent on every seen
        task. This is the metric that accounts for forgetting.
        """
        if not per_task_success:
            return None
        ap = float(np.mean(list(per_task_success.values())))
        self._last_ap = ap
        payload = {"global_step": self.global_step, "AP": ap}
        for task_name, sr in per_task_success.items():
            payload[f"eval_success_{task_name}"] = float(sr)
        if self.run is not None:
            self.run.log(payload)
        return ap

    def log_metrics(self, metrics, step=None, prefix=None):
        if not metrics:
            return {}

        global_step = self.global_step if step is None else int(step)
        payload = {"global_step": global_step}
        for key, value in metrics.items():
            if value is None:
                continue
            metric_key = f"{prefix}/{key}" if prefix else key
            payload[metric_key] = value

        if len(payload) > 1 and self.run is not None:
            self.run.log(payload)
        return payload

    def log_algorithm_metrics(self, metrics, step=None):
        return self.log_metrics(metrics, step=step)

    def log_video(self, task_name: str, frames: list):
        if self.run is None:
            return
        video = np.stack(frames)
        # DummyVecEnv may return (1, H, W, C) per frame → stack gives (T, 1, H, W, C)
        if video.ndim == 5:
            video = video[:, 0]
        # Grayscale envs return (H, W) per frame → stack gives (T, H, W); add channel dim
        if video.ndim == 3:
            video = video[..., np.newaxis]
        if video.ndim != 4:
            raise ValueError(f"Unexpected video shape after normalisation: {video.shape}")
        # wandb.Video expects (T, C, H, W)
        video = video.transpose(0, 3, 1, 2)
        self.run.log({
            f"eval/video_{task_name}": wandb.Video(video.astype(np.uint8), fps=30, format="mp4"),
            "global_step": self.global_step,
        })

    def finish(self):
        if self.run is None:
            return
        final_regret = self.regret_sum / self.regret_steps if self.regret_steps > 0 else 0.0
        self.run.summary["final_regret"] = float(final_regret)
        self.run.summary["total_global_steps"] = int(self.global_step)
        if self._last_ap is not None:
            self.run.summary["final_AP"] = float(self._last_ap)
        for task_name, rate in self._completed_task_diag.items():
            self.run.summary[f"final_diag_success_{task_name}"] = float(rate)
        self.run.finish()
