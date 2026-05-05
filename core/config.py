
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class OverlapInferenceConfig:
    """Configuration for overlapping patch inference to reduce boundary artifacts.

    This config enables overlapping patch-based prediction where:
    - Each patch extracts a larger region than the effective output
    - Only the center (effective) region is used for final stitching
    - Gaussian or average blending smooths overlapping regions

    Key parameters:
        enabled: Enable/disable overlapping inference
        patch_size: Full patch size extracted from image (e.g., [64, 64, 64])
        effective_size: Center region kept for output (e.g., [48, 48, 48])
        stride: Step size between patches (should equal effective_size for seamless coverage)
        padding_mode: How to handle image boundaries ("reflect", "constant", "none")
        padding_value: Fill value for constant padding
        blend_mode: How to combine overlapping predictions ("gaussian", "average")
        gaussian_sigma: Sigma for Gaussian weighting (relative to effective_size)
        use_amp: Enable automatic mixed precision (FP16) for memory efficiency
        batch_size: Number of patches to process simultaneously
    """
    enabled: bool = True
    patch_size: tuple[int, int, int] = (64, 64, 64)
    effective_size: tuple[int, int, int] = (48, 48, 48)
    padding_mode: str = "reflect"
    padding_value: float = 0.0
    blend_mode: str = "gaussian"
    gaussian_sigma: float = 0.4
    binarize_threshold: float = 0.5  # Threshold for converting probability to binary mask
    use_amp: bool = True
    batch_size: int = 4

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.enabled:
            if len(self.patch_size) != 3 or len(self.effective_size) != 3:
                raise ValueError("patch_size and effective_size must be 3-tuples")

            for ps, es in zip(self.patch_size, self.effective_size):
                if ps <= 0 or es <= 0:
                    raise ValueError("patch_size and effective_size must be positive")
                if es > ps:
                    raise ValueError(
                        f"effective_size ({es}) cannot exceed patch_size ({ps})"
                    )

            if self.blend_mode not in ("gaussian", "average"):
                raise ValueError(f"blend_mode must be 'gaussian' or 'average', got '{self.blend_mode}'")

            if self.padding_mode not in ("reflect", "constant", "none"):
                raise ValueError(
                    f"padding_mode must be 'reflect', 'constant', or 'none', "
                    f"got '{self.padding_mode}'"
                )

            if self.gaussian_sigma <= 0:
                raise ValueError(f"gaussian_sigma must be positive, got {self.gaussian_sigma}")

            if self.binarize_threshold < 0 or self.binarize_threshold > 1:
                raise ValueError(f"binarize_threshold must be in [0, 1], got {self.binarize_threshold}")

            if self.batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OverlapInferenceConfig":
        """Create config from dictionary, handling YAML list-to-tuple conversion."""
        if not data:
            return cls(enabled=False)

        # Convert list values to tuples for patch_size and effective_size
        kwargs = dict(data)
        if "patch_size" in kwargs and isinstance(kwargs["patch_size"], list):
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        if "effective_size" in kwargs and isinstance(kwargs["effective_size"], list):
            kwargs["effective_size"] = tuple(kwargs["effective_size"])

        return cls(**kwargs)


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
    # Segmentation: BCE + surface term. surface ∈ {"dice","tversky","focal_dice"}
    segmentation_surface_loss: str = "tversky"
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_dice_gamma: float = 4.0 / 3.0
    loss_smooth: float = 1e-6

    def segmentation_loss_kwargs(self) -> dict[str, float | str]:
        """Keyword arguments for ``script.eval_utils.combined_loss*``."""
        return {
            "surface": self.segmentation_surface_loss,
            "tversky_alpha": self.tversky_alpha,
            "tversky_beta": self.tversky_beta,
            "focal_dice_gamma": self.focal_dice_gamma,
            "smooth": self.loss_smooth,
        }


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
    label_suffix: str | list[str] = "_label.nii.gz"
    train_ratio: float = 0.8
    max_load: int = 1000


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
    use_pos_weight: bool = True
    early_stopping_patience: int = 10000
    early_stopping_min_delta: float = 1e-4
    threshold_search_steps: int = 19
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
    """Complete configuration for the Gradulate CTA segmentation pipeline.

    Attributes:
        model: Neural network architecture settings
        train: Training hyperparameters (epochs, batch_size, learning_rate, etc.)
        data: Data loading and preprocessing settings
        async_load: Memory management for large 3D volumes
        compile: PyTorch compile optimization settings
        checkpoint: Model checkpoint saving/loading configuration
        risk: Risk prediction task settings (CTA + tabular features)
        device: Compute device ("cuda" or "cpu")
        seed: Random seed for reproducibility
        log_dir: Directory for training logs
        eval_interval: Evaluation frequency during training
        overlap_inference: Overlapping patch inference configuration for reduced boundary artifacts
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    async_load: AsyncLoadConfig = field(default_factory=AsyncLoadConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    overlap_inference: OverlapInferenceConfig = field(default_factory=OverlapInferenceConfig)
    device: str = "cuda"
    seed: int = 42
    log_dir: str = "logs"
    eval_interval: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from a dictionary (e.g., parsed from YAML).

        Args:
            data: Dictionary with configuration parameters. Nested configs
                  are automatically converted to their respective dataclasses.

        Returns:
            Populated Config instance
        """
        model_data = data.pop("model", {})
        train_data = data.pop("train", {})
        data_data = data.pop("data", {})
        async_data = data.pop("async_load", {})
        compile_data = data.pop("compile", {})
        checkpoint_data = data.pop("checkpoint", {})
        risk_data = data.pop("risk", {})
        overlap_data = data.pop("overlap_inference", {})

        return cls(
            model=ModelConfig(**model_data),
            train=TrainConfig(**train_data),
            data=DataConfig(**data_data),
            async_load=AsyncLoadConfig(**async_data),
            compile=CompileConfig(**compile_data),
            checkpoint=CheckpointConfig(**checkpoint_data),
            risk=RiskConfig(**risk_data),
            overlap_inference=OverlapInferenceConfig.from_dict(overlap_data),
            **{k: v for k, v in data.items() if k not in [
                "model", "train", "data", "async_load", "compile", "checkpoint", "risk",
                "overlap_inference"
            ]}
        )

