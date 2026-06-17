import numpy as np
import wandb
import time


class ContinualLogger:
    def __init__(self, project, config, run_name=None, enable_wandb=True, entity=None, num_envs=1):
        self.enable_wandb = enable_wandb
        self.num_envs = int(num_envs)
        self.global_step = 0
        self.regret_sum = 0.0
        self.run = None
        self.start_time = time.time()
        # per-task episodic success tracking (reset each task)
        self._task_episodes = 0
        self._task_successes = 0
        # frozen snapshot of each completed task's final episodic success rate
        self._completed_task_aps: dict = {}
        if self.enable_wandb:
            self.run = wandb.init(project=project, entity=entity, name=run_name, config=config)
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step")
            self.run.define_metric("regret_running", summary="last")
            self.run.define_metric("train_task_episodic_success", summary="last")
            self.run.define_metric("train_ap_completed_tasks", summary="last")

    @staticmethod
    def step_success(successes):
        return float(np.mean(np.asarray(successes, dtype=np.float32)))

    def update_online(self, task_name, task_idx, episode_idx, timestep_in_episode, successes):
        online_success = self.step_success(successes)
        self.global_step += self.num_envs
        # weight by num_envs so the running average is over env-steps, matching global_step
        self.regret_sum += (1.0 - online_success) * self.num_envs
        regret_running = self.regret_sum / self.global_step

        payload = {
            "global_step": self.global_step,
            "task_idx": int(task_idx),
            "episode_idx": int(episode_idx),
            "timestep_in_episode": int(timestep_in_episode),
            "regret_running": float(regret_running),
            f"train_success_{task_name}": online_success,
        }
        if self.run is not None:
            self.run.log(payload)
        return online_success, float(regret_running)

    def on_episode_end(self, task_name: str, success: bool):
        """
        Call once per finished episode during training on the current task.
        Tracks episodic success rate and logs it continuously.
        """
        self._task_episodes += 1
        if success:
            self._task_successes += 1
        rate = self._task_successes / self._task_episodes
        payload = {
            "global_step": self.global_step,
            "train_task_episodic_success": float(rate),
            f"train_episodic_success_{task_name}": float(rate),
        }
        if self.run is not None:
            self.run.log(payload)
        return float(rate)

    def on_task_end(self, task_name: str):
        """
        Call once after all training episodes for a task are done.
        Freezes that task's episodic success rate, updates the running
        average over all completed tasks, then resets per-task counters.
        """
        final_rate = self._task_successes / self._task_episodes if self._task_episodes > 0 else 0.0
        self._completed_task_aps[task_name] = final_rate
        ap_completed = float(np.mean(list(self._completed_task_aps.values())))
        payload = {
            "global_step": self.global_step,
            f"train_final_episodic_success_{task_name}": final_rate,
            "train_ap_completed_tasks": ap_completed,
            "num_completed_tasks": len(self._completed_task_aps),
        }
        if self.run is not None:
            self.run.log(payload)
        # reset for the next task
        self._task_episodes = 0
        self._task_successes = 0
        return final_rate, ap_completed

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

    def log_video(self, task_name: str, ep_idx: int, frames: list):
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
            f"eval/video_{task_name}_ep{ep_idx}": wandb.Video(video.astype(np.uint8), fps=30, format="mp4"),
            "global_step": self.global_step,
        })

    def finish(self):
        if self.run is None:
            return
        final_regret = self.regret_sum / self.global_step if self.global_step > 0 else 0.0
        self.run.summary["final_regret_running"] = float(final_regret)
        self.run.summary["total_global_steps"] = int(self.global_step)
        if self._completed_task_aps:
            self.run.summary["final_train_ap_completed_tasks"] = float(
                np.mean(list(self._completed_task_aps.values()))
            )
            for task_name, rate in self._completed_task_aps.items():
                self.run.summary[f"final_train_episodic_success_{task_name}"] = float(rate)
        self.run.finish()
