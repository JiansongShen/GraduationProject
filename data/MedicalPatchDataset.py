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
from data.patch_sampler import create_sampler
from data.patch_utils import crop_patch
from data.preprocessing import apply_preprocessing, build_preprocess_config
from data.resample_helper import ResampleHelper, ResampleMeta
from data.spatial_utils import normalize_image
from utils.helper import find_all_file_paths_recursively


@dataclass(frozen=True)
class CaseRecord:
    image_path: str
    label_path: str


class MedicalPatchDataset(TorchDataset):
    """核心化简版：单数据对加载 + 自动重采样 + patch采样。"""

    def __init__(
        self,
        cfg: DataConfig,
        patch_size: tuple[int, int, int] = (128, 128, 128),
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        seed: int = 42,
    ):
        self.cfg = cfg
        self.patch_size = tuple(int(v) for v in patch_size)
        self.target_spacing = tuple(float(v) for v in target_spacing)
        self.patches_per_volume = max(1, int(cfg.patches_per_volume))
        self.patch_sampling_mode = cfg.patch_sampling_mode.lower()
        self.background_per_foreground = max(0, int(cfg.background_per_foreground))
        self.preprocess_cfg = build_preprocess_config(getattr(cfg, "preprocess", None))
        self.validate_geometry = bool(getattr(cfg, "validate_geometry", False))
        self.rng = random.Random(seed)
        self.resample = ResampleHelper(self.target_spacing)

        self._validate_sampling_mode(self.patch_sampling_mode)
        self.cases = self._build_case_records(cfg.train_dirs)
        if not self.cases:
            raise ValueError("No valid image-label pairs found in train_dirs")
        if self.validate_geometry:
            self._validate_case_geometries()

    def _validate_sampling_mode(self, mode: str) -> None:
        valid = {"foreground_priority", "foreground_only", "sequential"}
        if mode not in valid:
            raise ValueError(f"Unsupported patch_sampling_mode={mode}, expected one of {sorted(valid)}")

    def _find_label_path(self, image_path: str) -> str | None:
        image_prefix = image_path[: -len(self.cfg.origin_suffix)]
        for label_suffix in self.cfg.label_suffixes:
            candidate = image_prefix + label_suffix
            if os.path.exists(candidate):
                return candidate
        return None

    def _build_case_records(self, train_dirs: list[str]) -> list[CaseRecord]:
        records: list[CaseRecord] = []
        max_load = int(getattr(self.cfg, "max_load", 1000))
        skipped_missing_label = 0

        for root_dir in train_dirs:
            for path in find_all_file_paths_recursively(root_dir):
                if not path.endswith(self.cfg.origin_suffix):
                    continue
                label_path = self._find_label_path(path)
                if label_path is None:
                    skipped_missing_label += 1
                    logging.warning(
                        "Skipping case without label: image=%s expected_label_suffixes=%s",
                        path,
                        self.cfg.label_suffixes,
                    )
                    continue
                records.append(CaseRecord(image_path=path, label_path=label_path))
                if len(records) >= max_load:
                    if skipped_missing_label:
                        logging.warning(
                            "Skipped %d image(s) without labels while building dataset.",
                            skipped_missing_label,
                        )
                    return records
        if skipped_missing_label:
            logging.warning(
                "Skipped %d image(s) without labels while building dataset.",
                skipped_missing_label,
            )
        return records

    def _validate_case_geometries(self) -> None:
        for case in self.cases:
            image_itk = sitk.ReadImage(case.image_path)
            label_itk = sitk.ReadImage(case.label_path)
            image_geometry = (image_itk.GetSize(), image_itk.GetSpacing(), image_itk.GetOrigin(), image_itk.GetDirection())
            label_geometry = (label_itk.GetSize(), label_itk.GetSpacing(), label_itk.GetOrigin(), label_itk.GetDirection())
            if image_geometry != label_geometry:
                raise ValueError(
                    "Image/label geometry mismatch before preprocessing: "
                    f"image={case.image_path} label={case.label_path} "
                    f"image_geometry={image_geometry} label_geometry={label_geometry}"
                )

            image_train, _ = self.resample.to_train_space(image_itk, is_label=False)
            label_train, _ = self.resample.to_train_space(label_itk, is_label=True)
            train_geometry = (image_train.GetSize(), image_train.GetSpacing(), image_train.GetOrigin(), image_train.GetDirection())
            label_train_geometry = (label_train.GetSize(), label_train.GetSpacing(), label_train.GetOrigin(), label_train.GetDirection())
            if train_geometry != label_train_geometry:
                raise ValueError(
                    "Image/label geometry mismatch after resampling: "
                    f"image={case.image_path} label={case.label_path} "
                    f"image_geometry={train_geometry} label_geometry={label_train_geometry}"
                )

    def _load_case(self, case: CaseRecord) -> tuple[np.ndarray, np.ndarray, ResampleMeta]:
        image_itk = sitk.ReadImage(case.image_path)
        label_itk = sitk.ReadImage(case.label_path)

        image_train, meta = self.resample.to_train_space(image_itk, is_label=False)
        label_train, _ = self.resample.to_train_space(label_itk, is_label=True)

        image_np = sitk.GetArrayFromImage(image_train).astype(np.float32)
        label_np = sitk.GetArrayFromImage(label_train).astype(np.int64)
        image_np = apply_preprocessing(image_np, self.preprocess_cfg)
        image_np = normalize_image(image_np)
        return image_np, label_np, meta

    def _sample_starts(self, image: np.ndarray, label: np.ndarray, mode: str, patches_per_volume: int) -> list[tuple[int, int, int]]:
        sampler = create_sampler(
            mode=mode,
            patch_size=self.patch_size,
            rng=self.rng,
            bg_per_fg=self.background_per_foreground,
        )
        starts = sampler.sample(image, label, patches_per_volume)
        if not starts and mode in {"foreground_priority", "foreground_only"}:
            starts = create_sampler(
                mode="sequential",
                patch_size=self.patch_size,
                rng=self.rng,
                bg_per_fg=self.background_per_foreground,
            ).sample(image, label, patches_per_volume)
        return starts

    def _extract_patches(self, image: np.ndarray, label: np.ndarray, starts: list[tuple[int, int, int]]) -> tuple[Tensor, Tensor]:
        if not starts:
            raise ValueError("No patch starts sampled; check patch_size/sampling mode")

        image_patches: list[torch.Tensor] = []
        label_patches: list[torch.Tensor] = []
        for start in starts:
            img_patch = crop_patch(image, start, self.patch_size).astype(np.float32)
            lbl_patch = crop_patch(label, start, self.patch_size).astype(np.int64)
            image_patches.append(torch.from_numpy(img_patch[None, ...]))
            label_patches.append(torch.from_numpy(lbl_patch))

        return torch.stack(image_patches, dim=0), torch.stack(label_patches, dim=0)

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.get_patches(index)

    def get_patches(
        self,
        index: int,
        sampling_mode: Optional[str] = None,
        patches_per_volume: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        case = self.cases[index % len(self.cases)]
        image, label, _ = self._load_case(case)

        mode = (sampling_mode or self.patch_sampling_mode).lower()
        self._validate_sampling_mode(mode)
        ppv = int(patches_per_volume or self.patches_per_volume)

        starts = self._sample_starts(image, label, mode, ppv)
        image_patches, label_patches = self._extract_patches(image, label, starts)
        foreground_voxels = float((label_patches > 0).sum().item())
        total_voxels = float(label_patches.numel())
        logging.info(
            "Loaded patches | case=%s | mode=%s | patches=%d | foreground_ratio=%.8f",
            Path(case.image_path).name,
            mode,
            int(label_patches.shape[0]),
            foreground_voxels / total_voxels if total_voxels > 0.0 else 0.0,
        )
        return image_patches, label_patches

    def get_src_item(self, batch_idx: int) -> tuple[Tensor, Tensor]:
        image, label, _ = self._load_case(self.cases[batch_idx])
        return torch.from_numpy(image), torch.from_numpy(label)

    def get_src_label_path(self, batch_idx: int) -> Path:
        return Path(self.cases[batch_idx].label_path)

    def restore_to_original_space(self, pred_train_space: sitk.Image, case_idx: int, *, is_label: bool) -> sitk.Image:
        src_image = sitk.ReadImage(self.cases[case_idx].image_path)
        meta = ResampleMeta(
            original_spacing=src_image.GetSpacing(),
            original_size=src_image.GetSize(),
            original_origin=src_image.GetOrigin(),
            original_direction=src_image.GetDirection(),
            train_spacing=self.target_spacing,
        )
        return self.resample.to_original_space(pred_train_space, meta, is_label=is_label)
