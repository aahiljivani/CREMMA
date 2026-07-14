import numpy as np
import wandb
import time
from collections import deque


class ContinualLogger:
    def __init__(self, project, config, run_name=None, enable_wandb=True, entity=None,
                 num_envs=1, success_window=20):
        self.enable_wandb = enable_wandb
        self.num_envs = int(num_envs)
        self.global_step = 0
        self.regret_sum = 0.0
        self.regret_steps = 0
        self._last_ap = None
        self.run = None
        self.start_time = time.time()
        # Continual Bench's info["success"] is an instantaneous, episode-terminating
        # indicator, so online metrics are estimated from *episodic* success instead.
        self._ep_len = np.zeros(self.num_envs, dtype=np.int64)
        # rolling window of recent episode outcomes for the current task's learning curve
        self._success_window = deque(maxlen=int(success_window))
        self._num_tasks_done = 0
        # frozen diagonal p_tau(tau): learning perf, NOT AP
        self._completed_task_diag: dict = {}
        if self.enable_wandb:
            self.run = wandb.init(project=project, entity=entity, name=run_name, config=config)
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step")
            self.run.define_metric("regret_running", summary="last")
            self.run.define_metric("AP", summary="last")

    def update_online(self, task_name, task_idx, successes, dones):
        """Per-timestep CRL online metrics estimated from *episodic* success.

        Continual Bench terminates an episode the moment info["success"] is True,
        so the raw per-step flag is 0 for nearly every step; averaging it would pin
        regret at ~1. Instead we credit each completed episode with its binary
        outcome, which recovers a success rate in [0, 1].
        """
        successes = np.asarray(successes, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        self.global_step += self.num_envs
        self._ep_len += 1

        for i in np.nonzero(dones)[0]:
            ep_success = float(successes[i])  # 1.0 iff the episode terminated via success
            ep_len = int(self._ep_len[i])
            # Regret, Eq. 9: time-weighted area between the success curve and oracle=1.
            self.regret_sum += (1.0 - ep_success) * ep_len
            self.regret_steps += ep_len
            # Windowed online success rate -> smooth learning curve (paper Figure 5).
            self._success_window.append(ep_success)
            self._ep_len[i] = 0

        regret_running = self.regret_sum / self.regret_steps if self.regret_steps > 0 else 0.0
        train_final_success = (
            float(np.mean(self._success_window)) if self._success_window else 0.0
        )

        payload = {
            "global_step": self.global_step,
            "task_idx": int(task_idx),
            "regret_running": float(regret_running),
            f"train_final_success_{task_name}": float(train_final_success),
        }
        if self.run is not None:
            self.run.log(payload)
        return float(train_final_success), float(regret_running)

    def on_task_end(self, task_name: str):
        """Reset per-task online accumulators once a task's training is finished."""
        self._num_tasks_done += 1
        payload = {
            "global_step": self.global_step,
            "num_completed_tasks": self._num_tasks_done,
        }
        if self.run is not None:
            self.run.log(payload)
        self._success_window.clear()
        self._ep_len[:] = 0

    def log_offline_ap(self, per_task_success: dict, current_task: str = None):
        """
        Average Performance AP(w), Eq. 8: mean over all seen tasks of the CURRENT
        policy's offline success rate on each task. `per_task_success` is
        {task_name: success_rate} from evaluating the current agent on every seen
        task. This is the metric that accounts for forgetting.

        When `current_task` is given, freeze the diagonal p_tau(tau) (learning
        performance) from its offline eval right after learning it.
        """
        if not per_task_success:
            return None
        ap = float(np.mean(list(per_task_success.values())))
        self._last_ap = ap
        if current_task is not None and current_task in per_task_success:
            self._completed_task_diag[current_task] = float(per_task_success[current_task])
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

    def log_video(self, task_name: str, frames: list, key_prefix: str = "eval/video"):
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
            f"{key_prefix}_{task_name}": wandb.Video(video.astype(np.uint8), fps=30, format="mp4"),
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
