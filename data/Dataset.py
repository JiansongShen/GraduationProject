from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset

from core.config import DataConfig
from data.data_preprocesser import NiftiImage, resample_in_memory
from utils.helper import find_all_file_paths_recursively


@dataclass(frozen=True)
class CaseRecord:
    """A single training case composed of an image volume and an optional label volume.

    In medical imaging, one case usually corresponds to one patient scan or one exam.
    Keeping the loader case-oriented allows us to read only one volume at a time,
    which is the key idea behind the low-memory streaming strategy.
    """

    image_path: str
    label_path: Optional[str] = None


class MedicalPatchDataset(TorchDataset):
    """Low-memory 3D medical image dataset with case-wise streaming and online patch sampling.

    Design goals
    ------------
    1. Never keep the whole dataset in RAM.
    2. Load only one case when a sample is requested.
    3. Resample and crop patches on the fly.
    4. Support training on large 3D CT/MRI volumes even when memory is limited.

    This dataset follows "strategy A":
    - load one case;
    - preprocess it in memory;
    - sample one or more training patches from it;
    - release it immediately after the sample is returned.

    Notes
    -----
    - Images are converted to `float32` tensors.
    - Labels are converted to `long` tensors when provided.
    - The returned tensors are shaped as `[C, D, H, W]`.
    """

    def __init__(
        self,
        cfg: DataConfig,
        patch_size: tuple[int, int, int] = (128, 128, 128),
        target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        patches_per_volume: int = 1,
        foreground_sampling_prob: float = 0.5,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.patches_per_volume = max(1, patches_per_volume)
        self.foreground_sampling_prob = float(np.clip(foreground_sampling_prob, 0.0, 1.0))
        self.rng = random.Random(seed)

        # Build the case list from the configured training directories.
        # Each image file is matched with its corresponding label file if present.
        self.cases: list[CaseRecord] = self._build_case_records(cfg.train_dirs)
        if not self.cases:
            raise ValueError("No training cases were found. Please check `train_dirs` and `file_patterns`.")

    def _build_case_records(self, train_dirs: list[str]) -> list[CaseRecord]:
        """Collect all image files and infer label paths.

        The loader assumes a naming convention where an image file has a known suffix,
        for example `*_origin.nii.gz` or `*_brainpre.nii.gz`, and the label file can be
        obtained by replacing that suffix with `cfg.label_suffix`.
        """
        records: list[CaseRecord] = []
        seen_images: set[str] = set()

        for root_dir in train_dirs:
            for path in find_all_file_paths_recursively(root_dir):
                if not any(path.endswith(pattern.replace("*", "")) for pattern in self.cfg.file_patterns):
                    continue
                if path in seen_images:
                    continue
                seen_images.add(path)

                label_path = self._infer_label_path(path)
                if label_path is not None and not os.path.exists(label_path):
                    label_path = None

                records.append(CaseRecord(image_path=path, label_path=label_path))

        return records

    def _infer_label_path(self, image_path: str) -> Optional[str]:
        """Infer the label path from the image path.

        This is a lightweight convention-based matcher. It keeps the implementation
        simple and avoids loading the whole dataset into memory just to build pairs.
        """
        for pattern in self.cfg.file_patterns:
            suffix = pattern.replace("*", "")
            if image_path.endswith(suffix):
                return image_path[: -len(suffix)] + self.cfg.label_suffix
        return None

    def __len__(self) -> int:
        # Each case can contribute multiple patches per epoch.
        # This makes the effective length larger than the raw case count and
        # helps the model see more crops from each 3D volume.
        return len(self.cases) * self.patches_per_volume

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor | None]:
        """Load one case, sample one patch, and return it immediately.

        The dataset is intentionally streaming-oriented:
        - load the case from disk;
        - preprocess in RAM;
        - generate a patch;
        - discard the full volume after returning the sample.
        """
        case = self.cases[index % len(self.cases)]
        image, label = self._load_case(case)
        image_patch, label_patch = self._sample_patch(image, label)
        return image_patch, label_patch

    def _load_case(self, case: CaseRecord) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Load and preprocess one 3D case.

        We first read the image with SimpleITK, then resample it in memory to a
        consistent voxel spacing. A consistent spacing is important for 3D medical
        training because it stabilizes the physical receptive field across scans.
        """
        image_itk: NiftiImage = sitk.ReadImage(case.image_path)
        image_itk = resample_in_memory(image_itk, self.target_spacing, is_mask=False)
        image_np = sitk.GetArrayFromImage(image_itk).astype(np.float32)

        label_np: Optional[np.ndarray] = None
        if case.label_path is not None:
            label_itk: NiftiImage = sitk.ReadImage(case.label_path)
            label_itk = resample_in_memory(label_itk, self.target_spacing, is_mask=True)
            label_np = sitk.GetArrayFromImage(label_itk).astype(np.int64)

        return image_np, label_np

    def _sample_patch(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
    ) -> tuple[Tensor, Tensor | None]:
        """Sample one spatial patch from the current case.

        If a label is available, we optionally bias the crop toward foreground voxels.
        This is a common strategy in medical segmentation because lesions and organs
        often occupy only a small fraction of the full volume.
        """
        image = self._normalize_image(image)
        image = self._ensure_min_shape(image)
        if label is not None:
            label = self._ensure_min_shape(label)

        start = self._choose_patch_start(label)
        patch_image = self._crop_patch(image, start)
        patch_image = torch.from_numpy(patch_image[None, ...].astype(np.float32))

        patch_label_tensor: Tensor | None = None
        if label is not None:
            patch_label = self._crop_patch(label, start)
            patch_label_tensor = torch.from_numpy(patch_label.astype(np.int64))

        return patch_image, patch_label_tensor

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Apply a simple robust normalization.

        This implementation uses z-score normalization on the entire case.
        If your dataset contains strong outliers, you can replace this with a
        percentile-based clip + normalization policy.
        """
        image = image.astype(np.float32, copy=False)
        mean = float(image.mean())
        std = float(image.std())
        if std < 1e-8:
            return image - mean
        return (image - mean) / std

    def _ensure_min_shape(self, array: np.ndarray) -> np.ndarray:
        """Pad the volume if it is smaller than the requested patch size.

        Padding is done with zeros so that even very small scans can still produce
        a valid training patch without crashing the loader.
        """
        target_d, target_h, target_w = self.patch_size
        d, h, w = array.shape

        pad_d = max(0, target_d - d)
        pad_h = max(0, target_h - h)
        pad_w = max(0, target_w - w)

        if pad_d == pad_h == pad_w == 0:
            return array

        pad_before = (pad_d // 2, pad_h // 2, pad_w // 2)
        pad_after = (
            pad_d - pad_before[0],
            pad_h - pad_before[1],
            pad_w - pad_before[2],
        )

        return np.pad(
            array,
            ((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1]), (pad_before[2], pad_after[2])),
            mode="constant",
            constant_values=0,
        )

    def _choose_patch_start(self, label: Optional[np.ndarray]) -> tuple[int, int, int]:
        """Choose the top-left-front corner of a patch.

        When foreground labels exist, we sample around foreground voxels with a
        certain probability to increase the chance of seeing positive examples.
        Otherwise we fall back to uniform random sampling.
        """
        if label is not None and label.any() and self.rng.random() < self.foreground_sampling_prob:
            foreground_coords = np.argwhere(label > 0)
            center_z, center_y, center_x = foreground_coords[self.rng.randrange(len(foreground_coords))]
            return self._center_to_start((int(center_z), int(center_y), int(center_x)), label.shape)

        max_z = max(0, label.shape[0] - self.patch_size[0]) if label is not None else 0
        max_y = max(0, label.shape[1] - self.patch_size[1]) if label is not None else 0
        max_x = max(0, label.shape[2] - self.patch_size[2]) if label is not None else 0

        if label is None:
            # If no label is available, use the image shape via a dummy padded crop space.
            return (0, 0, 0)

        return (
            self.rng.randint(0, max_z) if max_z > 0 else 0,
            self.rng.randint(0, max_y) if max_y > 0 else 0,
            self.rng.randint(0, max_x) if max_x > 0 else 0,
        )

    def _center_to_start(self, center: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[int, int, int]:
        """Convert a foreground center point into a valid crop start position."""
        cz, cy, cx = center
        max_z = max(0, shape[0] - self.patch_size[0])
        max_y = max(0, shape[1] - self.patch_size[1])
        max_x = max(0, shape[2] - self.patch_size[2])

        start_z = min(max(cz - self.patch_size[0] // 2, 0), max_z)
        start_y = min(max(cy - self.patch_size[1] // 2, 0), max_y)
        start_x = min(max(cx - self.patch_size[2] // 2, 0), max_x)
        return start_z, start_y, start_x

    def _crop_patch(self, array: np.ndarray, start: tuple[int, int, int]) -> np.ndarray:
        """Extract a `[D, H, W]` patch from a 3D array."""
        sz, sy, sx = start
        dz, dy, dx = self.patch_size
        return array[sz : sz + dz, sy : sy + dy, sx : sx + dx]
