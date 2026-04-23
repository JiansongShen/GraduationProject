import random
from pathlib import Path

import numpy as np
import torch

from core.config import Config
from core.config_loader import load_config


class SystemSetting:
    @staticmethod
    def set_seed(seed: int) -> None:
        """设置所有随机种子以确保可重复性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __init__(self, seed: int,
                 config_path: Path = Path(__file__).resolve().parent / "config" / "config.yaml") -> None:
        self.set_seed(seed)
        self.global_cfg = load_config(config_path)

    @staticmethod
    def get_gpu_memory() -> dict[str, float]:
        """get gpu information but just applicable for single card"""
        if not torch.cuda.is_available():
            return {
                "allocated": 0.0,
                "total": 0.0,
                "reserved": 0.0,
                "free": 0.0,
            }
        return {
            "allocated": torch.cuda.memory_allocated(),
            "total": torch.cuda.get_device_properties(0).total_memory,
            "reserved": torch.cuda.memory_reserved(),
            "free": torch.cuda.memory_reserved(),
        }

    @staticmethod
    def init_sys_default(cls):
        cls.set_seed(42)

    def get_cfg(self) -> Config:
        return self.global_cfg