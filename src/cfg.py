from pathlib import Path

from omegaconf import OmegaConf


def parse_cfg(cfg_path: Path):
    base = OmegaConf.load(cfg_path / "default.yaml")
    cli = OmegaConf.from_cli()

    policy = cli.get("policy")
    if policy is None:
        raise ValueError("policy must be specified via CLI, e.g. policy=DDPG")

    policy_cfg = cfg_path / "algorithms" / f"{policy.lower()}.yaml"
    if policy_cfg.exists():
        base.merge_with(OmegaConf.load(policy_cfg))

    mode = cli.get("benchmark_mode", base.benchmark_mode)
    mode_cfg = cfg_path / f"{mode}.yaml"
    if mode_cfg.exists():
        base.merge_with(OmegaConf.load(mode_cfg))

    for k, v in cli.items():
        if v is None:
            cli[k] = True
    base.merge_with(cli)

    if base.benchmark_mode == "task":
        if base.single_task_name not in base.task_list:
            raise ValueError(f"single_task_name must be one of {list(base.task_list)}")

    return base
