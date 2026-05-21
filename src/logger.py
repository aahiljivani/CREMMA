import numpy as np
import wandb
import time


class ContinualLogger:
    def __init__(self, project, config, run_name=None, enable_wandb=True, entity=None):
        self.enable_wandb = enable_wandb
        self.global_step = 0
        self.regret_sum = 0.0
        self.run = None
        self.start_time = time.time()
        if self.enable_wandb:
            self.run = wandb.init(project=project, entity=entity, name=run_name, config=config)
            self.run.define_metric("global_step")
            self.run.define_metric("*", step_metric="global_step")
            self.run.define_metric("regret_running", summary="last")
            self.run.define_metric("eval_ap_seen_tasks", summary="max")

    @staticmethod
    def step_success(successes):
        return float(np.mean(np.asarray(successes, dtype=np.float32)))

    def update_online(self, task_name, task_idx, episode_idx, timestep_in_episode, successes):
        online_success = self.step_success(successes)
        self.global_step += 1
        self.regret_sum += (1.0 - online_success)
        regret_running = self.regret_sum / self.global_step

        payload = {
            "global_step": self.global_step,
            "task_idx": int(task_idx),
            "episode_idx": int(episode_idx),
            "timestep_in_episode": int(timestep_in_episode),
            "online_success_current_task": online_success,
            "regret_running": float(regret_running),
            f"train_success_{task_name}": online_success,
        }
        if self.run is not None:
            self.run.log(payload)
        return online_success, float(regret_running)

    @staticmethod
    def average_performance(task_scores):
        if not task_scores:
            return 0.0
        return float(np.mean(list(task_scores.values())))

    def log_evaluation(self, seen_tasks, task_scores):
        ap_seen = self.average_performance(task_scores)
        payload = {
            "global_step": self.global_step,
            "num_seen_tasks": len(seen_tasks),
            "eval_ap_seen_tasks": ap_seen,
        }
        for task_name, score in task_scores.items():
            payload[f"eval_success_{task_name}"] = float(score)
        if self.run is not None:
            self.run.log(payload)
        return ap_seen

    def finish(self, final_task_scores=None):
        if self.run is None:
            return
        final_task_scores = final_task_scores or {}
        final_ap = self.average_performance(final_task_scores)
        final_regret = self.regret_sum / self.global_step if self.global_step > 0 else 0.0
        self.run.summary["final_eval_ap_seen_tasks"] = float(final_ap)
        self.run.summary["final_regret_running"] = float(final_regret)
        self.run.summary["total_global_steps"] = int(self.global_step)
        for task_name, score in final_task_scores.items():
            self.run.summary[f"final_eval_success_{task_name}"] = float(score)
        self.run.finish()
    
    def sac_log(self):
        pass
