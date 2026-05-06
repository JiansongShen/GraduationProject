"""Patch sampling strategies for 3D medical image segmentation.

Supports multiple sampling modes:
- sequential: Grid-based patch extraction
- foreground_priority: Foreground patches with background ratio control
- foreground_only: Only foreground patches
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class PatchSampler(ABC):
    """Base class for patch sampling strategies."""

    def __init__(self, patch_size: tuple[int, int, int], rng: random.Random):
        self.patch_size = patch_size
        self.rng = rng
        self.stride: Optional[tuple[int, int, int]] = None

    @abstractmethod
    def sample(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
        num_patches: int,
    ) -> list[tuple[int, int, int]]:
        """Return list of (z, y, x) patch start coordinates."""
        ...

    def iter_grid_starts(
        self,
        shape: tuple[int, int, int],
        stride: Optional[tuple[int, int, int]] = None,
    ) -> list[tuple[int, int, int]]:
        """Generate grid-based patch starts with configurable stride.

        Args:
            shape: Volume shape (D, H, W)
            stride: Step size for grid iteration. Defaults to self.stride or patch_size.
                   When stride < patch_size, generates overlapping positions.
        """
        pd, ph, pw = self.patch_size
        s = stride if stride is not None else (self.stride if self.stride else self.patch_size)
        sd, sh, sw = s
        d, h, w = shape
        starts = []
        for z in range(0, d, sd):
            for y in range(0, h, sh):
                for x in range(0, w, sw):
                    starts.append((z, y, x))
        return starts

    def is_foreground_patch(
        self, label: np.ndarray, start: tuple[int, int, int]
    ) -> bool:
        """Check if patch contains any foreground voxels."""
        z, y, x = start
        pd, ph, pw = self.patch_size
        patch_label = label[z:z + pd, y:y + ph, x:x + pw]
        return bool((patch_label > 0).any())


class SequentialSampler(PatchSampler):
    """Grid-based sequential sampling.

    Args:
        patch_size: Patch 尺寸 (D, H, W)
        rng: 随机数生成器
        stride: 步长, 默认等于 patch_size (非重叠)
            - 设置为有效区域大小时支持重叠推理
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        rng: random.Random,
        stride: Optional[tuple[int, int, int]] = None,
    ):
        super().__init__(patch_size, rng)
        self.stride = stride if stride is not None else patch_size

    def sample(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
        num_patches: int,
    ) -> list[tuple[int, int, int]]:
        return self.iter_grid_starts(image.shape)[:num_patches]

    def iter_grid_starts(self, shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        """Generate grid-based patch starts with configurable stride.

        当 stride < patch_size 时, 会产生重叠的 patch 位置。
        """
        pd, ph, pw = self.patch_size
        sd, sh, sw = self.stride
        d, h, w = shape
        starts = []
        for z in range(0, d, sd):
            for y in range(0, h, sh):
                for x in range(0, w, sw):
                    starts.append((z, y, x))
        return starts


class ForegroundSampler(PatchSampler):
    """Foreground-centered sampling with configurable background ratio.

    Args:
        patch_size: Patch 尺寸 (D, H, W)
        rng: 随机数生成器
        bg_per_fg: 背景 patch 与前景 patch 的比例
        stride: 步长, 支持重叠推理
    """

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        rng: random.Random,
        bg_per_fg: int = 0,
        stride: Optional[tuple[int, int, int]] = None,
    ):
        super().__init__(patch_size, rng)
        self.bg_per_fg = bg_per_fg
        self.stride = stride if stride is not None else patch_size

    def sample(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
        num_patches: int,
    ) -> list[tuple[int, int, int]]:
        if label is None:
            logging.warning("No label provided, falling back to sequential")
            return self.iter_grid_starts(image.shape)[:num_patches]

        fg_coords = np.argwhere(label > 0)
        if fg_coords.size == 0:
            logging.warning("No foreground voxels found, falling back to sequential")
            return self.iter_grid_starts(image.shape)[:num_patches]

        selected: list[tuple[int, int, int]] = []
        bg_starts = [s for s in self.iter_grid_starts(image.shape) if not self.is_foreground_patch(label, s)]
        bg_cursor = 0

        max_fg_patches = max(1, num_patches - min(num_patches - 1, self.bg_per_fg))
        for _ in range(min(max_fg_patches, len(fg_coords))):
            idx = self.rng.randrange(len(fg_coords))
            cz, cy, cx = (int(v) for v in fg_coords[idx])
            start = self._foreground_center_to_start((cz, cy, cx), image.shape)
            selected.append(start)

            for _ in range(self.bg_per_fg):
                if len(selected) >= num_patches or bg_cursor >= len(bg_starts):
                    break
                selected.append(bg_starts[bg_cursor])
                bg_cursor += 1

        while len(selected) < num_patches:
            idx = self.rng.randrange(len(fg_coords))
            cz, cy, cx = (int(v) for v in fg_coords[idx])
            selected.append(self._foreground_center_to_start((cz, cy, cx), image.shape))

        return selected[:num_patches]

    def _foreground_center_to_start(
        self,
        center: tuple[int, int, int],
        shape: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        cz, cy, cx = center
        jitter = tuple(
            self.rng.randint(-size // 4, size // 4 - 1)
            for size in self.patch_size
        )
        patch_center = (cz + jitter[0], cy + jitter[1], cx + jitter[2])
        max_start = tuple(max(0, dim - size) for dim, size in zip(shape, self.patch_size))
        return tuple(
            int(np.clip(coord - size // 2, 0, limit))
            for coord, size, limit in zip(patch_center, self.patch_size, max_start)
        )


class ForegroundOnlySampler(PatchSampler):
    """Only sample patches containing foreground."""

    def sample(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
        num_patches: int,
    ) -> list[tuple[int, int, int]]:
        if label is None:
            return self.iter_grid_starts(image.shape)[:num_patches]

        grid_starts = self.iter_grid_starts(image.shape)
        fg_starts = [s for s in grid_starts if self.is_foreground_patch(label, s)]

        if not fg_starts:
            logging.warning("No foreground patches found, falling back to sequential")
            return grid_starts[:num_patches]
        return fg_starts[:num_patches]


class RandomSampler(PatchSampler):
    """Completely random sampling within valid bounds."""

    def sample(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
        num_patches: int,
    ) -> list[tuple[int, int, int]]:
        selected: list[tuple[int, int, int]] = []
        d, h, w = image.shape
        pd, ph, pw = self.patch_size

        seen = set()
        attempts = 0
        max_attempts = num_patches * 4

        while len(selected) < num_patches and attempts < max_attempts:
            attempts += 1
            sz = self.rng.randint(0, max(0, d - pd))
            sy = self.rng.randint(0, max(0, h - ph))
            sx = self.rng.randint(0, max(0, w - pw))
            key = (sz // 8, sy // 8, sx // 8)
            if key in seen:
                continue
            seen.add(key)
            selected.append((sz, sy, sx))

        return selected


class ForegroundBiasedSampler(PatchSampler):
    """Random sampling biased toward foreground with probability."""

    def __init__(self, patch_size: tuple[int, int, int], rng: random.Random, fg_prob: float = 0.5):
        super().__init__(patch_size, rng)
        self.fg_prob = fg_prob

    def sample(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray],
        num_patches: int,
    ) -> list[tuple[int, int, int]]:
        selected: list[tuple[int, int, int]] = []
        d, h, w = image.shape
        pd, ph, pw = self.patch_size

        for _ in range(num_patches):
            if label is not None and label.any() and self.rng.random() < self.fg_prob:
                fg_coords = np.argwhere(label > 0)
                if fg_coords.size > 0:
                    idx = self.rng.randrange(len(fg_coords))
                    cz, cy, cx = fg_coords[idx]
                    start = self._center_to_start((int(cz), int(cy), int(cx)), image.shape)
                    selected.append(start)
                    continue

            # Random fallback
            sz = self.rng.randint(0, max(0, d - pd))
            sy = self.rng.randint(0, max(0, h - ph))
            sx = self.rng.randint(0, max(0, w - pw))
            selected.append((sz, sy, sx))

        return selected

    def _center_to_start(
        self, center: tuple[int, int, int], shape: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        cz, cy, cx = center
        max_z = max(0, shape[0] - self.patch_size[0])
        max_y = max(0, shape[1] - self.patch_size[1])
        max_x = max(0, shape[2] - self.patch_size[2])
        return (
            min(max(cz - self.patch_size[0] // 2, 0), max_z),
            min(max(cy - self.patch_size[1] // 2, 0), max_y),
            min(max(cx - self.patch_size[2] // 2, 0), max_x),
        )


def create_sampler(
    mode: str,
    patch_size: tuple[int, int, int],
    rng: random.Random,
    bg_per_fg: int = 0,
    fg_prob: float = 0.5,
    stride: Optional[tuple[int, int, int]] = None,
) -> PatchSampler:
    """Factory function to create appropriate sampler.

    Args:
        mode: 采样模式 ("sequential", "foreground_priority", "foreground_only")
        patch_size: Patch 尺寸
        rng: 随机数生成器
        bg_per_fg: 背景与前景 patch 比例
        fg_prob: 前景采样概率
        stride: 步长, 支持重叠推理 (默认等于 patch_size)
    """
    mode = mode.lower()
    if mode == "sequential":
        return SequentialSampler(patch_size, rng, stride=stride)
    elif mode == "foreground_priority":
        return ForegroundSampler(patch_size, rng, bg_per_fg=bg_per_fg, stride=stride)
    elif mode == "foreground_only":
        return ForegroundOnlySampler(patch_size, rng)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
