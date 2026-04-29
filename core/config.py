
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
    patches_per_volume: int = 64
    train_dirs: list[str] = field(default_factory=list)
    eval_dirs: list[str] = field(default_factory=list)
    patch_sampling_mode: str = "sequential"
    background_per_foreground: int = 2
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
    """风险预测任务配置（CTA + 表格特征）"""

    enabled: bool = False
    excel_path: str = "/home/napbad/project/graduate_proj/core_model/dataset/data.xlsx"
    label_column: str = "破裂"
    id_column: str = "7lesion_id"
    feature_columns: list[str] = field(default_factory=list)
    task_type: str = "binary"
    save_dir: str = "checkpoints/risk"
    use_cleaned_data: bool = True
    use_dummy_metadata: bool = False
    dummy_num_rows: int = 512
    metadata_columns: list[str] = field(default_factory=lambda: ["年龄", "性别", "部位"])
    num_heads: int = 4
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    auto_max_features: int = 64
    feature_select_method: str = "mutual_info"
    categorical_min_frequency: int = 5
    use_pyradiomics_overlap_only: bool = True
    clinic_columns: list[str] = field(
        default_factory=lambda: ["性别", "年龄", "高血压", "心脏病", "糖尿病", "脑血管硬化", "饮酒", "抽烟", "出血史"]
    )
    saved_data_dir: str = "data/saved_data"
    predict_res_path: str = ""
    src_path: str = ""
    radiomics_backend: str = "pyradiomics-cuda"
    cta_viewer_mode: str = "tri-planar"
    segmentation_endpoint: str = "/api/cta/segment"
    risk_endpoint: str = "/api/risk/predict"


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

