import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    name: str = "unet"
    in_channels: int = 1
    out_channels: int = 1
    base_filters: int = 64
    depth: int = 4
    dropout: float = 0.0
    norm_type: str = "instance"
    activation: str = "relu"


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patch_size: tuple[int, int, int] = (64, 64, 64)
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    loss_type: str = "tversky_bce"
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_gamma: float = 2.0
    loss_smooth: float = 1e-6

    def segmentation_loss_kwargs(self) -> dict[str, float | str]:
        return {
            "loss_type": self.loss_type,
            "tversky_alpha": self.tversky_alpha,
            "tversky_beta": self.tversky_beta,
            "focal_gamma": self.focal_gamma,
            "smooth": self.loss_smooth,
        }


@dataclass
class DataConfig:
    train_dirs: list[str] = field(default_factory=list)
    eval_dirs: list[str] = field(default_factory=list)
    origin_suffix: str = "_origin.nii.gz"
    label_suffixes: list[str] = field(default_factory=lambda: ["_label.nii.gz"])
    patches_per_volume: int = 64  # Maximum patches sampled per volume.
    patch_sampling_mode: str = "foreground_priority"
    background_per_foreground: int = 2
    max_load: int = 1000
    eval_max_load: int | None = None
    validate_geometry: bool = False
    eval_max_load: int | None = None
    preprocess: dict[str, Any] = field(default_factory=dict)

    @property
    def label_suffix(self) -> str:
        return self.label_suffixes[0] if self.label_suffixes else "_label.nii.gz"


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_best: bool = True
    save_interval: int = 10


@dataclass
class RiskConfig:
    enabled: bool = False
    riskdataset_path: str = ""
    label_column: str = "label"
    clinic_columns: list[str] = field(default_factory=list)
    saved_data_dir: str = "data/saved_data"
    feature_columns_output: str = "risk_feature_columns.json"
    save_dir: str = "checkpoints/risk"
    hidden_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    use_pos_weight: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    threshold_search_steps: int = 19


@dataclass
class InferenceConfig:
    patch_size: tuple[int, int, int] = (64, 64, 64)
    effective_size: tuple[int, int, int] = (48, 48, 48)
    batch_size: int = 4
    use_amp: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    device: str = "cuda"
    seed: int = 42
    log_dir: str = "logs"
    eval_every_n_epochs: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        payload = dict(data)
        model_data = payload.pop("model", {})
        train_data = payload.pop("train", {})
        data_data = payload.pop("data", {})
        checkpoint_data = payload.pop("checkpoint", {})
        risk_data = payload.pop("risk", {})
        inference_data = payload.pop("inference", {})
        overlap_data = payload.pop("overlap_inference", {})
        payload.pop("async_load", None)
        payload.pop("compile", None)

        if "eval_interval" in payload and "eval_every_n_epochs" not in payload:
            payload["eval_every_n_epochs"] = payload.pop("eval_interval")
        else:
            payload.pop("eval_interval", None)

        if overlap_data and not inference_data:
            inference_data = {
                "patch_size": overlap_data.get("patch_size"),
                "effective_size": overlap_data.get("effective_size"),
                "batch_size": overlap_data.get("batch_size"),
                "use_amp": overlap_data.get("use_amp"),
            }

        train_data = _normalize_train_data(train_data)
        data_data = _normalize_data_data(data_data)
        checkpoint_data = _filter_dataclass_kwargs(CheckpointConfig, checkpoint_data)
        risk_data = _normalize_risk_data(risk_data)
        inference_data = _normalize_inference_data(inference_data)
        payload = _filter_dataclass_kwargs(cls, payload, field_names={"device", "seed", "log_dir", "eval_every_n_epochs"})

        return cls(
            model=ModelConfig(**_filter_dataclass_kwargs(ModelConfig, model_data)),
            train=TrainConfig(**train_data),
            data=DataConfig(**data_data),
            checkpoint=CheckpointConfig(**checkpoint_data),
            risk=RiskConfig(**risk_data),
            inference=InferenceConfig(**inference_data),
            **payload,
        )


def _filter_dataclass_kwargs(dataclass_type: type, values: dict[str, Any], field_names: set[str] | None = None) -> dict[str, Any]:
    allowed = field_names or {field.name for field in dataclasses.fields(dataclass_type)}
    return {key: value for key, value in values.items() if key in allowed}


def _normalize_train_data(train_data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(train_data)
    if isinstance(normalized.get("patch_size"), list):
        normalized["patch_size"] = tuple(normalized["patch_size"])
    if "segmentation_surface_loss" in normalized and "loss_type" not in normalized:
        surface = str(normalized.pop("segmentation_surface_loss")).lower()
        normalized["loss_type"] = "focal_bce" if "focal" in surface else "tversky_bce"
    if "focal_dice_gamma" in normalized and "focal_gamma" not in normalized:
        normalized["focal_gamma"] = normalized.pop("focal_dice_gamma")
    normalized.pop("max_patches_per_volume", None)
    return _filter_dataclass_kwargs(TrainConfig, normalized)


def _normalize_data_data(data_data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data_data)
    label_suffixes = normalized.get("label_suffixes")
    label_suffix = normalized.pop("label_suffix", None)
    if label_suffixes is None:
        if isinstance(label_suffix, list):
            label_suffixes = [str(item) for item in label_suffix if str(item)]
        elif label_suffix:
            label_suffixes = [str(label_suffix)]
    if label_suffixes is not None:
        normalized["label_suffixes"] = [str(item) for item in label_suffixes if str(item)]

    preprocess = normalized.get("preprocess")
    if isinstance(preprocess, list):
        normalized["preprocess"] = {"enabled": True, "steps": preprocess}
    elif isinstance(preprocess, str):
        normalized["preprocess"] = {"enabled": True, "steps": [preprocess]}
    elif preprocess is None:
        normalized["preprocess"] = {}

    origin_suffix = normalized.get("origin_suffix")
    file_patterns = normalized.get("file_patterns")
    if not origin_suffix and isinstance(file_patterns, list) and file_patterns:
        pattern = str(file_patterns[0])
        if pattern.startswith("*"):
            normalized["origin_suffix"] = pattern[1:]
    return _filter_dataclass_kwargs(DataConfig, normalized)


def _normalize_risk_data(risk_data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(risk_data)
    if "riskdataset_path" not in normalized and "excel_path" in normalized:
        normalized["riskdataset_path"] = normalized["excel_path"]
    if "feature_columns_output" not in normalized:
        normalized["feature_columns_output"] = RiskConfig.feature_columns_output
    return _filter_dataclass_kwargs(RiskConfig, normalized)


def _normalize_inference_data(inference_data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(inference_data)
    for key in ("patch_size", "effective_size"):
        if key in normalized and isinstance(normalized[key], list):
            normalized[key] = tuple(normalized[key])
    return _filter_dataclass_kwargs(InferenceConfig, normalized)
