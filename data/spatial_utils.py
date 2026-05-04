"""Spatial transformation utilities for 3D medical images.

Provides functions for resampling labels to match image geometry,
normalization, and volume shape estimation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk

if TYPE_CHECKING:
    from data.data_preprocesser import NiftiImage


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Z-score normalization on the entire volume."""
    image = image.astype(np.float32, copy=False)
    mean = float(image.mean())
    std = float(image.std())
    if std < 1e-8:
        return image - mean
    return (image - mean) / std


def estimate_new_shape(
    old_shape: tuple[int, int, int],
    old_spacing: tuple[float, float, float],
    new_spacing: tuple[float, float, float],
) -> tuple[int, int, int]:
    """Estimate volume shape after resampling to new spacing."""
    old_shape = np.array(old_shape)
    old_spacing = np.array(old_spacing)
    new_spacing = np.array(new_spacing)
    return tuple(np.round(old_shape * old_spacing / new_spacing).astype(int))


def estimate_volume_memory_mb(shape: tuple[int, int, int], dtype=np.float32) -> float:
    """Estimate memory footprint of a volume."""
    bytes_per_voxel = np.dtype(dtype).itemsize
    return np.prod(shape) * bytes_per_voxel / 1024 / 1024


def compute_physical_bounds(itk_img: sitk.Image) -> tuple[float, float, float, float, float, float]:
    """Compute physical world bounds from corner voxels."""
    sz = itk_img.GetSize()
    corners_phys = [
        itk_img.TransformIndexToPhysicalPoint((0, 0, 0)),
        itk_img.TransformIndexToPhysicalPoint((sz[0] - 1, 0, 0)),
        itk_img.TransformIndexToPhysicalPoint((0, sz[1] - 1, 0)),
        itk_img.TransformIndexToPhysicalPoint((0, 0, sz[2] - 1)),
        itk_img.TransformIndexToPhysicalPoint((sz[0] - 1, sz[1] - 1, sz[2] - 1)),
    ]
    all_x = [c[0] for c in corners_phys]
    all_y = [c[1] for c in corners_phys]
    all_z = [c[2] for c in corners_phys]
    return (min(all_x), max(all_x), min(all_y), max(all_y), min(all_z), max(all_z))


def check_overlap(
    a_min: float, a_max: float, b_min: float, b_max: float
) -> str:
    """Check physical-space overlap between two ranges."""
    lo, hi = max(a_min, b_min), min(a_max, b_max)
    return "NONE" if hi <= lo else f"{lo:.1f}~{hi:.1f} (width={hi-lo:.1f}mm)"


def resample_label_to_image_geometry(
    label_itk: sitk.Image,
    ref_image: sitk.Image,
) -> sitk.Image:
    """Resample label volume to match reference image's voxel grid.

    Applies a physical-space affine transform to map reference voxels
    to their corresponding label voxels, then resamples the label
    onto the reference grid using Nearest-Neighbor interpolation.
    """
    label_phys = compute_physical_bounds(label_itk)
    ref_phys = compute_physical_bounds(ref_image)

    ov_x = check_overlap(label_phys[0], label_phys[1], ref_phys[0], ref_phys[1])
    ov_y = check_overlap(label_phys[2], label_phys[3], ref_phys[2], ref_phys[3])
    ov_z = check_overlap(label_phys[4], label_phys[5], ref_phys[4], ref_phys[5])

    if ov_x == "NONE" or ov_y == "NONE" or ov_z == "NONE":
        logging.error("Label and image have no overlap in physical space")

    label_dir = np.array(label_itk.GetDirection()).reshape(3, 3)
    ref_dir = np.array(ref_image.GetDirection()).reshape(3, 3)
    label_dir /= np.linalg.norm(label_dir, axis=0, keepdims=True)
    ref_dir /= np.linalg.norm(ref_dir, axis=0, keepdims=True)

    A = np.linalg.inv(label_dir) @ ref_dir @ np.diag(
        np.array(ref_image.GetSpacing()) / np.array(label_itk.GetSpacing())
    )
    b = (
        np.linalg.inv(label_dir)
        @ (np.array(ref_image.GetOrigin()) - np.array(label_itk.GetOrigin()))
        / np.array(label_itk.GetSpacing())
    ).astype(float)

    transform = sitk.AffineTransform(3)
    transform.SetMatrix(A.T.flatten())
    transform.SetTranslation(b.tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(ref_image.GetSpacing())
    resampler.SetSize(ref_image.GetSize())
    resampler.SetOutputDirection(ref_image.GetDirection())
    resampler.SetOutputOrigin(ref_image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    result = resampler.Execute(label_itk)

    label_arr = sitk.GetArrayFromImage(label_itk)
    result_arr = sitk.GetArrayFromImage(result)
    logging.debug(
        "label_resample fg_before=%d fg_after=%d",
        int((label_arr > 0).sum()),
        int((result_arr > 0).sum()),
    )
    return result


def check_label_geometry_compatibility(
    label_spacing: tuple[float, ...],
    img_spacing: tuple[float, ...],
    label_size: tuple[int, ...],
    img_size: tuple[int, ...],
) -> bool:
    """Check if label and image geometry are significantly different."""
    spacing_ratio = tuple(
        max(s1, s2) / min(s1, s2) if min(s1, s2) > 1e-6 else 0.0
        for s1, s2 in zip(label_spacing, img_spacing)
    )
    size_ratio = tuple(
        max(s1, s2) / min(s1, s2) if min(s1, s2) > 0 else 0.0
        for s1, s2 in zip(label_size, img_size)
    )
    return max(spacing_ratio) <= 2.0 and max(size_ratio) <= 3.0
