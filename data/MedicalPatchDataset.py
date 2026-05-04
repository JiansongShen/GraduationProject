"""Low-memory 3D medical image dataset with case-wise streaming and online patch sampling."""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from core.config import DataConfig
from data.data_preprocesser import NiftiImage, resample_in_memory
from data.patch_sampler import create_sampler
from data.patch_utils import crop_patch
from data.spatial_utils import (
    check_label_geometry_compatibility,
    normalize_image,
    resample_label_to_image_geometry,
)
from utils.helper import find_all_file_paths_recursively


@dataclass(frozen=True)
class CaseRecord:
    """A single training case composed of an image volume and an optional label volume."""
    image_path: str
    label_path: Optional[str] = None


class MedicalPatchDataset(TorchDataset):
    """Low-memory 3D medical image dataset with streaming and online patch sampling.

    Design goals:
        1. Never keep the whole dataset in RAM
        2. Load only one case when a sample is requested
        3. Resample and crop patches on the fly
        4. Support training on large 3D CT/MRI volumes

    Returns:
        Tensors shaped as [P, C, D, H, W] for images and [P, D, H, W] for labels
    """

    def __init__(
        self,
        cfg: DataConfig,
        patch_size: tuple[int, int, int] = (128, 128, 128),
        target_spacing: tuple[float, float, float] = (1, 1, 1),
        foreground_sampling_prob: float = 0.5,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.patches_per_volume = max(1, cfg.patches_per_volume)
        self.rng = random.Random(seed)
        self.patch_sampling_mode = cfg.patch_sampling_mode.lower()
        self.background_per_foreground = max(0, int(cfg.background_per_foreground))
        self.foreground_sampling_prob = float(np.clip(foreground_sampling_prob, 0.0, 1.0))
        self._validate_sampling_mode()

        self.cases: list[CaseRecord] = self._build_case_records(cfg.train_dirs)
        if len(self.cases) > self.cfg.max_load:
            self.cases = self.cases[: self.cfg.max_load]

        if not self.cases:
            raise ValueError("No training cases found. Check `train_dirs` and `file_patterns`.")

    def _validate_sampling_mode(self) -> None:
        valid = {"sequential", "foreground_priority", "foreground_only"}
        if self.patch_sampling_mode not in valid:
            raise ValueError(
                f"Unsupported patch_sampling_mode: {self.patch_sampling_mode}. "
                f"Supported: {sorted(valid)}"
            )

    # ── Case Loading ──────────────────────────────────────────────────────────

    def _build_case_records(self, train_dirs: list[str]) -> list[CaseRecord]:
        """Collect all image files and infer corresponding label paths."""
        records: list[CaseRecord] = []
        seen: set[str] = set()

        for root_dir in train_dirs:
            for path in find_all_file_paths_recursively(root_dir):
                if not self._matches_file_pattern(path):
                    continue
                if path in seen:
                    continue
                seen.add(path)
                label_path = self._infer_label_path(path)
                records.append(CaseRecord(image_path=path, label_path=label_path))

        return records

    def _matches_file_pattern(self, path: str) -> bool:
        return any(path.endswith(p.replace("*", "")) for p in self.cfg.file_patterns)

    def _label_suffixes(self) -> list[str]:
        suffixes = self.cfg.label_suffix
        return [suffixes] if isinstance(suffixes, str) else list(suffixes)

    def _infer_label_path(self, image_path: str) -> Optional[str]:
        """Find first existing label file from configured suffix candidates."""
        for pattern in self.cfg.file_patterns:
            suffix = pattern.replace("*", "")
            if not image_path.endswith(suffix):
                continue
            image_prefix = image_path[: -len(suffix)]
            for label_suffix in self._label_suffixes():
                candidate = image_prefix + label_suffix
                if os.path.exists(candidate):
                    return candidate
        return None

    def _load_case(self, case: CaseRecord) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Load and preprocess one 3D case: resample to target spacing."""
        image_itk: NiftiImage = sitk.ReadImage(case.image_path)
        image_itk = resample_in_memory(image_itk, self.target_spacing, is_mask=False)
        image_np = sitk.GetArrayFromImage(image_itk).astype(np.float32)

        label_np: Optional[np.ndarray] = None
        if case.label_path is not None:
            if not os.path.exists(case.label_path):
                logging.warning("Label file not found: %s", case.label_path)
            else:
                label_itk = sitk.ReadImage(case.label_path)
                if not check_label_geometry_compatibility(
                    label_itk.GetSpacing(),
                    image_itk.GetSpacing(),
                    label_itk.GetSize(),
                    image_itk.GetSize(),
                ):
                    logging.warning("Significant label/image geometry mismatch for: %s", case.image_path)
                label_itk = resample_label_to_image_geometry(label_itk, image_itk)
                label_np = sitk.GetArrayFromImage(label_itk).astype(np.int64)

        return image_np, label_np

    # ── Patch Extraction ──────────────────────────────────────────────────────

    def _extract_patches(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
        starts: list[tuple[int, int, int]],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract patches from pre-computed start positions."""
        image_patches, label_patches = [], []
        for start in starts:
            image_patches.append(
                torch.from_numpy(crop_patch(image, start, self.patch_size)[None, ...].astype(np.float32))
            )
            if label is not None:
                label_patches.append(
                    torch.from_numpy(crop_patch(label, start, self.patch_size).astype(np.int64))
                )

        if not image_patches:
            raise ValueError("No patches selected. Check patch_size and sampling_mode.")

        images_tensor = torch.stack(image_patches, dim=0)
        return images_tensor, torch.stack(label_patches, dim=0) if label_patches else None

    def _sample_starts(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
        mode: str,
    ) -> list[tuple[int, int, int]]:
        """Sample patch start positions based on mode."""
        sampler = create_sampler(
            mode=mode,
            patch_size=self.patch_size,
            rng=self.rng,
            bg_per_fg=self.background_per_foreground,
        )
        return sampler.sample(image, label, self.patches_per_volume)

    # ── Dataset Interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[Tensor, Optional[Tensor]]:
        return self.get_patches(index, sampling_mode=self.patch_sampling_mode)

    def get_patches(
        self,
        index: int,
        sampling_mode: Optional[str] = None,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Load one volume and return patches with explicit sampling mode.

        Use `sampling_mode="sequential"` for evaluation to ensure deterministic order.
        """
        if not self.cases:
            raise IndexError("Dataset is empty.")

        case_index = index % len(self.cases)
        case = self.cases[case_index]

        image, label = self._load_case(case)
        image = normalize_image(image)

        mode = (sampling_mode or self.patch_sampling_mode).lower()

        if mode == "sequential":
            starts = self._sample_starts(image, label, mode)
        elif mode in {"foreground_priority", "foreground_only"}:
            starts = self._sample_starts(image, label, mode)
            if not starts and label is not None:
                logging.warning("Falling back to sequential for case: %s", case.image_path)
                starts = self._sample_starts(image, label, "sequential")
        else:
            starts = self._sample_starts(image, label, mode)

        return self._extract_patches(image, label, starts)

    # ── Source Data Access ────────────────────────────────────────────────────

    def get_src_item(self, batch_idx: int) -> tuple[Tensor, Optional[Tensor]]:
        """Get full (non-patched) image and label tensors."""
        image, label = self._load_case(self.cases[batch_idx])
        return torch.from_numpy(image), torch.from_numpy(label) if label is not None else None

    def get_src_label_path(self, batch_idx: int) -> Path:
        if self.cases[batch_idx].label_path is None:
            raise ValueError("Label path is None")
        return Path(self.cases[batch_idx].label_path)
