import dataclasses
import yaml

from pathlib import Path
from core.config import Config


def load_config(path: str | Path) -> Config:
    """加载YAML配置文件"""
    path_obj: Path = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return Config.from_dict(data)


def save_config(config: Config, path: str | Path) -> None:
    """保存配置到YAML文件"""
    path_obj: Path = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    data = {"model": dataclasses.asdict(config.model), "train": dataclasses.asdict(config.train),
            "data": dataclasses.asdict(config.data), "async_load": dataclasses.asdict(config.async_load),
            "compile": dataclasses.asdict(config.compile), "checkpoint": dataclasses.asdict(config.checkpoint),
            "risk": dataclasses.asdict(config.risk), "device": config.device, "seed": config.seed,
            "log_dir": config.log_dir, "eval_interval": config.eval_interval}

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
