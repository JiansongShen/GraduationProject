"""Patch manipulation utilities for 3D medical images."""

from __future__ import annotations

import numpy as np


def crop_patch(
    array: np.ndarray,
    start: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> np.ndarray:
    """Extract a 3D patch with safe out-of-bounds zero padding.

    Args:
        array: Source 3D array [D, H, W]
        start: Starting coordinate (z, y, x)
        patch_size: Desired patch size (d, h, w)

    Returns:
        Patched array of exactly patch_size with zeros for out-of-bounds regions
    """
    sz, sy, sx = start
    dz, dy, dx = patch_size
    src_d, src_h, src_w = array.shape

    ez, ey, ex = sz + dz, sy + dy, sx + dx

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


def pad_to_minimum_shape(
    array: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Pad array so every dimension can yield at least one full patch."""
    pad_width = []
    for dim_size, target_dim in zip(array.shape, target_shape):
        missing = max(0, target_dim - dim_size)
        pad_width.append((0, missing))
    if not any(after > 0 for _, after in pad_width):
        return array
    return np.pad(array, tuple(pad_width), mode="constant", constant_values=0)
