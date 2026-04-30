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
from triton.language import tensor

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


def _normalize_image(image: np.ndarray) -> np.ndarray:
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
        target_spacing: tuple[float, float, float] = (0.5, 0.5, 0.5),
        foreground_sampling_prob: float = 0.5,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.patch_size = patch_size
        self.target_spacing = target_spacing
        self.patches_per_volume = max(1, cfg.patches_per_volume)
        self.foreground_sampling_prob = float(np.clip(foreground_sampling_prob, 0.0, 1.0))
        self.rng = random.Random(seed)
        self.patch_sampling_mode = cfg.patch_sampling_mode.lower()
        self.background_per_foreground = max(0, int(cfg.background_per_foreground))
        valid_sampling_modes = {"sequential", "foreground_priority", "foreground_only"}
        if self.patch_sampling_mode not in valid_sampling_modes:
            raise ValueError(
                f"Unsupported patch_sampling_mode: {cfg.patch_sampling_mode}. "
                f"Supported modes: {sorted(valid_sampling_modes)}"
            )

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
            suffix = pattern.replace("*", ""    )
            if image_path.endswith(suffix):
                return image_path[: -len(suffix)] + self.cfg.label_suffix
        return None

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

    def _sample_single_patch(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
    ) -> tuple[Tensor, Tensor | None]:
        """Sample one spatial patch from the current case.

        If a label is available, we optionally bias the crop toward foreground voxels.
        This is a common strategy in medical segmentation because lesions and organs
        often occupy only a small fraction of the full volume.
        """
        start = self._choose_patch_start(image.shape, label)
        patch_image = self._crop_patch(image, start)
        patch_image = torch.from_numpy(patch_image[None, ...].astype(np.float32))

        patch_label_tensor: Tensor | None = None
        if label is not None:
            patch_label = self._crop_patch(label, start)
            patch_label_tensor = torch.from_numpy(patch_label.astype(np.int64))

        return patch_image, patch_label_tensor

    def __len__(self):
        # Dataset length represents how many volumes are indexable, not how many
        # random patches might be sampled from those volumes.
        return len(self.cases)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor | None]:
        """Load one volume and select patches with the configured sampling mode.

        `patches_per_volume` is a maximum cap, not a required sampling count.

        Shapes:
        - images: [P, C, D, H, W]
        - labels: [P, D, H, W] or None
        """
        return self.get_patches(index, sampling_mode=self.patch_sampling_mode)

    def get_patches(
        self,
        index: int,
        sampling_mode: str | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Load one volume and return patches with an explicit sampling mode.

        Evaluation should call this with `sampling_mode="sequential"` so the saved
        prediction volume is stitched with the same deterministic z/y/x order.
        """
        if len(self.cases) == 0:
            raise IndexError("Dataset is empty.")

        case_index = index % len(self.cases)
        case = self.cases[case_index]
        image, label = self._load_case(case)
        image = _normalize_image(image)

        patch_d, patch_h, patch_w = self.patch_size
        image = self._pad_to_minimum_patch_shape(image)
        if label is not None:
            label = self._pad_to_minimum_patch_shape(label)

        selected_starts = self._select_patch_starts(image, label, sampling_mode=sampling_mode)
        image_patches: list[Tensor] = []
        label_patches: list[Tensor] = []
        for z, y, x in selected_starts:
            patch_image = image[z:z + patch_d, y:y + patch_h, x:x + patch_w]
            image_patches.append(torch.from_numpy(patch_image[None, ...].astype(np.float32)))
            if label is not None:
                patch_label = label[z:z + patch_d, y:y + patch_h, x:x + patch_w]
                label_patches.append(torch.from_numpy(patch_label.astype(np.int64)))

        if not image_patches:
            raise ValueError(
                "No patches were selected for this case. Please check patch_size and patch_sampling_mode."
            )

        images_tensor = torch.stack(image_patches, dim=0)
        if label is None:
            return images_tensor, None
        return images_tensor, torch.stack(label_patches, dim=0)

    def _iter_patch_starts(self, image_shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        patch_d, patch_h, patch_w = self.patch_size
        image_d, image_h, image_w = image_shape
        starts: list[tuple[int, int, int]] = []
        for z in range(0, image_d - patch_d + 1, patch_d):
            for y in range(0, image_h - patch_h + 1, patch_h):
                for x in range(0, image_w - patch_w + 1, patch_w):
                    starts.append((z, y, x))
        return starts

    def _is_foreground_patch(self, label: np.ndarray, start: tuple[int, int, int]) -> bool:
        z, y, x = start
        patch_d, patch_h, patch_w = self.patch_size
        patch_label = label[z:z + patch_d, y:y + patch_h, x:x + patch_w]
        return bool((patch_label > 0).any())

    def _select_patch_starts(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
        sampling_mode: str | None = None,
    ) -> list[tuple[int, int, int]]:
        starts = self._iter_patch_starts(image.shape)
        if not starts:
            return []

        mode = (sampling_mode or self.patch_sampling_mode).lower()
        if mode not in {"sequential", "foreground_priority", "foreground_only"}:
            raise ValueError(f"Unsupported patch_sampling_mode: {mode}")

        if label is None or mode == "sequential":
            return starts[: self.patches_per_volume]

        foreground_starts: list[tuple[int, int, int]] = []
        background_starts: list[tuple[int, int, int]] = []
        for start in starts:
            if self._is_foreground_patch(label, start):
                foreground_starts.append(start)
            else:
                background_starts.append(start)

        if mode == "foreground_only":
            if not foreground_starts:
                logging.warning(
                    "No foreground patches found in case, falling back to sequential sampling for one case."
                )
                return starts[: self.patches_per_volume]
            return foreground_starts[: self.patches_per_volume]

        if mode == "foreground_priority":
            if not foreground_starts:
                logging.warning(
                    "No foreground patches found in case, falling back to sequential sampling for one case."
                )
                return starts[: self.patches_per_volume]

            selected: list[tuple[int, int, int]] = []
            bg_cursor = 0
            for fg_start in foreground_starts:
                if len(selected) >= self.patches_per_volume:
                    break
                selected.append(fg_start)

                for _ in range(self.background_per_foreground):
                    if len(selected) >= self.patches_per_volume:
                        break
                    if bg_cursor >= len(background_starts):
                        break
                    selected.append(background_starts[bg_cursor])
                    bg_cursor += 1

            if len(selected) < self.patches_per_volume:
                for fg_start in foreground_starts:
                    if len(selected) >= self.patches_per_volume:
                        break
                    if fg_start not in selected:
                        selected.append(fg_start)

            if len(selected) < self.patches_per_volume:
                for bg_start in background_starts:
                    if len(selected) >= self.patches_per_volume:
                        break
                    if bg_start not in selected:
                        selected.append(bg_start)
            return selected

        return starts[: self.patches_per_volume]

    def _pad_to_minimum_patch_shape(self, array: np.ndarray) -> np.ndarray:
        """Pad a volume so every dimension can yield at least one full patch."""
        pad_width = []
        for dim_size, patch_dim in zip(array.shape, self.patch_size):
            missing = max(0, patch_dim - dim_size)
            pad_width.append((0, missing))
        if not any(after > 0 for _, after in pad_width):
            return array
        return np.pad(array, tuple(pad_width), mode="constant", constant_values=0)

    def _choose_patch_start(self, image_shape: tuple[int, int, int], label: Optional[np.ndarray]) -> tuple[int, int, int]:
        """Choose the top-left-front corner of a patch.

        When foreground labels exist, we sample around foreground voxels with a
        certain probability to increase the chance of seeing positive examples.
        Otherwise we fall back to uniform random sampling.
        """
        if label is not None and label.any() and self.rng.random() < self.foreground_sampling_prob:
            foreground_coords = np.argwhere(label > 0)
            center_z, center_y, center_x = foreground_coords[self.rng.randrange(len(foreground_coords))]
            return self._center_to_start((int(center_z), int(center_y), int(center_x)), label.shape)

        max_z = max(0, image_shape[0] - self.patch_size[0])
        max_y = max(0, image_shape[1] - self.patch_size[1])
        max_x = max(0, image_shape[2] - self.patch_size[2])

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
        """Extract a `[D, H, W]` patch with safe out-of-bounds zero padding."""
        sz, sy, sx = start
        dz, dy, dx = self.patch_size
        src_d, src_h, src_w = array.shape

        ez = sz + dz
        ey = sy + dy
        ex = sx + dx

        src_z0 = max(0, sz)
        src_y0 = max(0, sy)
        src_x0 = max(0, sx)
        src_z1 = min(src_d, ez)
        src_y1 = min(src_h, ey)
        src_x1 = min(src_w, ex)

        patch = np.zeros((dz, dy, dx), dtype=array.dtype)
        if src_z1 <= src_z0 or src_y1 <= src_y0 or src_x1 <= src_x0:
            return patch

        dst_z0 = src_z0 - sz
        dst_y0 = src_y0 - sy
        dst_x0 = src_x0 - sx
        dst_z1 = dst_z0 + (src_z1 - src_z0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        patch[dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1] = array[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
        return patch

    def get_src_item(self, batch_idx: int) -> tuple[Tensor, Tensor | None]:
        image, label = self._load_case(self.cases[batch_idx])
        return Tensor(image), Tensor(label) if label is not None else None
