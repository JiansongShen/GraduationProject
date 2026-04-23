
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "unet"
    in_channels: int = 1
    out_channels: int = 1
    base_filters: int = 64
    depth: int = 4
    use_residual: bool = False
    use_attention: bool = False
    dropout: float = 0.0
    norm_type: str = "batch"  # "batch", "instance", "group"
    activation: str = "relu"  # "relu", "leaky_relu", "gelu"


@dataclass
class TrainConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patch_size: tuple[int, int, int] = (128, 128, 128)
    max_patches_per_volume: int | None = None
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 5


@dataclass
class DataConfig:
    """数据配置"""
    train_dirs: list[str] = field(default_factory=list)
    eval_dirs: list[str] = field(default_factory=list)
    num_workers: int = 4
    prefetch_factor: int = 2
    file_patterns: list[str] = field(
        default_factory=lambda: ["*_origin.nii.gz", "*_brainpre.nii.gz"]
    )
    label_suffix: str = "_label.nii.gz"
    train_ratio: float = 0.8


@dataclass
class AsyncLoadConfig:
    """异步动态加载配置"""
    ram_cache_gb: float = 8.0
    vram_batch_gb: float = 4.0
    auto_config: bool = True


@dataclass
class CompileConfig:
    """编译优化配置"""
    enabled: bool = False
    mode: str = "default"
    fullgraph: bool = False


@dataclass
class CheckpointConfig:
    """检查点配置"""
    save_dir: str = "checkpoints"
    save_best: bool = True
    save_interval: int = 10
    max_keep: int = 5


@dataclass
class RiskConfig:
    """动脉瘤破裂/高风险二分类（表格 + 可选形态特征）"""

    excel_path: str = "dataset/data.xlsx"
    label_column: str = "破裂"
    id_column: str = "7lesion_id"
    feature_columns: list[str] = field(default_factory=list)
    artifact_path: str = "checkpoints/risk_model.joblib"
    classifier: str = "hist_gradient_boosting"
    test_size: float = 0.2
    auto_max_features: int | None = 200


@dataclass
class Config:
    """完整配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    async_load: AsyncLoadConfig = field(default_factory=AsyncLoadConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    device: str = "cuda"
    seed: int = 42
    log_dir: str = "logs"
    eval_interval: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """从字典创建配置"""
        model_data = data.pop("model", {})
        train_data = data.pop("train", {})
        data_data = data.pop("data", {})
        async_data = data.pop("async_load", {})
        compile_data = data.pop("compile", {})
        checkpoint_data = data.pop("checkpoint", {})
        risk_data = data.pop("risk", {})

        return cls(
            model=ModelConfig(**model_data),
            train=TrainConfig(**train_data),
            data=DataConfig(**data_data),
            async_load=AsyncLoadConfig(**async_data),
            compile=CompileConfig(**compile_data),
            checkpoint=CheckpointConfig(**checkpoint_data),
            risk=RiskConfig(**risk_data),
            **{k: v for k, v in data.items() if k not in [
                "model", "train", "data", "async_load", "compile", "checkpoint", "risk"
            ]}
        )

