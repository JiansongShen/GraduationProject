from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage import exposure
from skimage.filters import frangi


@dataclass(frozen=True)
class PreprocessConfig:
    enabled: bool = False
    steps: list[str] = field(default_factory=list)
    clahe_kernel_size: tuple[int, int, int] | None = None
    clahe_clip_limit: float = 0.01
    clahe_nb_bins: int = 256
    frangi_scale_range: tuple[float, float] = (1.0, 10.0)
    frangi_scale_step: float = 2.0
    frangi_alpha: float = 0.5
    frangi_beta: float = 0.5
    frangi_gamma: float | None = None
    frangi_black_ridges: bool = False
    frangi_preserve_sign: bool = False
    output_blend_weight: float = 1.0


DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()


def build_preprocess_config(raw: dict[str, Any] | None) -> PreprocessConfig:
    if not raw:
        return DEFAULT_PREPROCESS_CONFIG

    normalized = dict(raw)
    steps = normalized.get("steps")
    if isinstance(steps, str):
        normalized["steps"] = [steps]
    elif isinstance(steps, (tuple, list)):
        normalized["steps"] = [str(step).strip().lower() for step in steps if str(step).strip()]

    for key in ("clahe_kernel_size", "frangi_scale_range"):
        value = normalized.get(key)
        if isinstance(value, list):
            normalized[key] = tuple(value)

    alias_pairs = {
        "clip_limit": "clahe_clip_limit",
        "kernel_size": "clahe_kernel_size",
        "nbins": "clahe_nb_bins",
        "sigmas": "frangi_scale_range",
        "scale_range": "frangi_scale_range",
    }
    for alias, target in alias_pairs.items():
        if alias in normalized and target not in normalized:
            normalized[target] = normalized.pop(alias)

    allowed = {field.name for field in PreprocessConfig.__dataclass_fields__.values()}
    filtered = {key: value for key, value in normalized.items() if key in allowed}
    return PreprocessConfig(**filtered)


def apply_preprocessing(image: np.ndarray, cfg: PreprocessConfig | None) -> np.ndarray:
    config = cfg or DEFAULT_PREPROCESS_CONFIG
    if not config.enabled or not config.steps:
        return image.astype(np.float32, copy=False)

    processed = image.astype(np.float32, copy=False)
    for step in config.steps:
        if step == "clahe":
            processed = _apply_clahe(processed, config)
        elif step in {"frangi", "frgnhi"}:
            processed = _apply_frangi(processed, config)
        elif step in {"none", "identity"}:
            continue
        else:
            raise ValueError(f"Unsupported preprocess step: {step}")
    return processed.astype(np.float32, copy=False)


def _apply_clahe(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    image_01 = _minmax_to_unit_interval(image)
    enhanced = exposure.equalize_adapthist(
        image_01,
        kernel_size=cfg.clahe_kernel_size,
        clip_limit=float(cfg.clahe_clip_limit),
        nbins=int(cfg.clahe_nb_bins),
    )
    return enhanced.astype(np.float32, copy=False)


def _apply_frangi(image: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    image_01 = _minmax_to_unit_interval(image)
    vesselness = frangi(
        image_01,
        sigmas=_build_sigmas(cfg.frangi_scale_range, cfg.frangi_scale_step),
        alpha=float(cfg.frangi_alpha),
        beta=float(cfg.frangi_beta),
        gamma=None if cfg.frangi_gamma is None else float(cfg.frangi_gamma),
        black_ridges=bool(cfg.frangi_black_ridges),
    ).astype(np.float32, copy=False)
    vesselness = _minmax_to_unit_interval(vesselness)
    if cfg.frangi_preserve_sign:
        source = _minmax_to_unit_interval(image)
        return ((1.0 - float(cfg.output_blend_weight)) * source + float(cfg.output_blend_weight) * vesselness).astype(np.float32)
    return vesselness


def _build_sigmas(scale_range: tuple[float, float], scale_step: float) -> np.ndarray:
    low = float(scale_range[0])
    high = float(scale_range[1])
    step = float(scale_step)
    if step <= 0.0:
        raise ValueError("frangi_scale_step must be > 0")
    if high < low:
        low, high = high, low
    count = max(1, int(np.floor((high - low) / step)) + 1)
    return np.asarray([low + idx * step for idx in range(count)], dtype=np.float32)


def _minmax_to_unit_interval(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    min_value = float(arr.min())
    max_value = float(arr.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - min_value) / (max_value - min_value)
