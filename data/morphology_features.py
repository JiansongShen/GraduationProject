"""Morphology feature extraction from segmentation masks.

This module extracts shape and morphological features from 3D medical image
segmentation masks (e.g., aneurysm masks) that are clinically relevant for
risk prediction models.

Features include:
- Basic geometric measurements (volume, surface area, dimensions)
- Shape descriptors (sphericity, elongation, compactness)
- Size ratios and derived metrics (AR, SR, etc.)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

logger = logging.getLogger(__name__)


def extract_morphology_features(
    image_path: str,
    mask_path: str | None = None,
) -> dict[str, float]:
    """Extract morphology features from a segmented medical image.

    Args:
        image_path: Path to the original 3D medical image (CTA/MRI).
        mask_path: Optional path to the segmentation mask. If None, features
                   will be extracted assuming the image itself is binary.

    Returns:
        Dictionary mapping feature names to their numeric values.
    """
    try:
        # Load image and mask
        image = sitk.ReadImage(image_path)
        if mask_path:
            mask = sitk.ReadImage(mask_path)
        else:
            # Assume image is already a binary mask
            mask = image

        # Convert to numpy arrays
        image_array = sitk.GetArrayFromImage(image).astype(np.float32)
        mask_array = sitk.GetArrayFromImage(mask).astype(np.int32)

        # Get spacing information
        spacing = image.GetSpacing()  # (x, y, z) in mm
        spacing_array = np.array(spacing[::-1])  # Convert to (z, y, x) for numpy

        # Extract features
        features = _compute_shape_features(mask_array, spacing_array, image_array)

        logger.info("Extracted %d morphology features from %s", len(features), mask_path or image_path)
        return features

    except Exception as e:
        logger.error("Failed to extract morphology features: %s", str(e))
        # Return default features to avoid breaking the pipeline
        return _get_default_features()


def _compute_shape_features(
    mask: np.ndarray,
    spacing: np.ndarray,
    image: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute comprehensive shape features from a binary mask.

    Args:
        mask: Binary segmentation mask (3D array).
        spacing: Voxel spacing in mm (z, y, x).
        image: Original image array for intensity-based features.

    Returns:
        Dictionary of morphology features.
    """
    features: dict[str, float] = {}

    # Ensure binary mask
    binary_mask = (mask > 0).astype(np.int32)

    # Check if mask has any foreground
    if binary_mask.sum() == 0:
        logger.warning("Empty mask detected, returning default features")
        return _get_default_features()

    # Find connected components
    labeled_mask, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return _get_default_features()

    # Use the largest connected component
    component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))
    largest_component_label = np.argmax(component_sizes) + 1
    largest_component = (labeled_mask == largest_component_label).astype(np.int32)

    # Basic volume measurements
    voxel_count = largest_component.sum()
    voxel_volume_mm3 = float(voxel_count * np.prod(spacing))
    features["Volume_mm3"] = voxel_volume_mm3

    # Bounding box
    bbox_slices = ndimage.find_objects(labeled_mask)[largest_component_label - 1]
    bbox_size_voxels = [s.stop - s.start for s in bbox_slices]
    bbox_size_mm = [size * sp for size, sp in zip(bbox_size_voxels, spacing)]

    features["BoundingBox_Length_mm"] = bbox_size_mm[0]  # Z dimension
    features["BoundingBox_Width_mm"] = bbox_size_mm[1]   # Y dimension
    features["BoundingBox_Depth_mm"] = bbox_size_mm[2]   # X dimension
    features["Max_Diameter_mm"] = max(bbox_size_mm)
    features["Min_Diameter_mm"] = min(bbox_size_mm)
    features["Mean_Diameter_mm"] = float(np.mean(bbox_size_mm))

    # Derived ratios (clinically important for aneurysms)
    height = bbox_size_mm[0]  # Assuming Z is height
    width = bbox_size_mm[1]
    depth = bbox_size_mm[2]

    # Aspect Ratio (AR) = Height / Width
    ar = height / width if width > 0 else 0.0
    features["Aspect_Ratio"] = ar

    # Size Ratio (SR) variants
    mean_diameter = features["Mean_Diameter_mm"]
    sr_depth_mean = depth / mean_diameter if mean_diameter > 0 else 0.0
    sr_max_mean = features["Max_Diameter_mm"] / mean_diameter if mean_diameter > 0 else 0.0

    features["SR_Depth_to_Mean"] = sr_depth_mean
    features["SR_Max_to_Mean"] = sr_max_mean

    # Surface area calculation
    surface_area_mm2 = _calculate_surface_area(largest_component, spacing)
    features["Surface_Area_mm2"] = surface_area_mm2

    # Surface-to-volume ratio
    sv_ratio = surface_area_mm2 / voxel_volume_mm3 if voxel_volume_mm3 > 0 else 0.0
    features["Surface_Volume_Ratio"] = sv_ratio

    # Sphericity (how close to a sphere)
    sphericity = _calculate_sphericity(voxel_volume_mm3, surface_area_mm2)
    features["Sphericity"] = sphericity

    # Elongation
    elongation = min(bbox_size_mm) / max(bbox_size_mm) if max(bbox_size_mm) > 0 else 1.0
    features["Elongation"] = elongation

    # Compactness
    compactness = _calculate_compactness(voxel_volume_mm3, surface_area_mm2)
    features["Compactness"] = compactness

    # Intensity-based features (if original image provided)
    if image is not None:
        intensity_features = _compute_intensity_features(image, largest_component)
        features.update(intensity_features)

    return features


def _calculate_surface_area(mask: np.ndarray, spacing: np.ndarray) -> float:
    """Calculate surface area using marching cubes approximation.

    Uses gradient-based method for surface area estimation.
    """
    # Compute gradients
    grad_z, grad_y, grad_x = np.gradient(mask.astype(np.float32))

    # Scale gradients by spacing
    grad_z /= spacing[0]
    grad_y /= spacing[1]
    grad_x /= spacing[2]

    # Calculate gradient magnitude
    grad_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)

    # Surface area is the sum of gradient magnitudes at the boundary
    # Multiply by voxel volume for proper units
    surface_area = float(np.sum(grad_magnitude) * np.prod(spacing))

    return surface_area


def _calculate_sphericity(volume_mm3: float, surface_area_mm2: float) -> float:
    """Calculate sphericity (1 = perfect sphere, <1 = irregular).

    Sphericity = (π^(1/3) * (6V)^(2/3)) / A
    where V is volume and A is surface area.
    """
    if surface_area_mm2 <= 0 or volume_mm3 <= 0:
        return 0.0

    equivalent_sphere_surface = (np.pi ** (1/3)) * ((6 * volume_mm3) ** (2/3))
    sphericity = equivalent_sphere_surface / surface_area_mm2

    # Clamp to [0, 1]
    return float(np.clip(sphericity, 0.0, 1.0))


def _calculate_compactness(volume_mm3: float, surface_area_mm2: float) -> float:
    """Calculate compactness metric.

    Compactness = V^(2/3) / A
    Higher values indicate more compact shapes.
    """
    if surface_area_mm2 <= 0 or volume_mm3 <= 0:
        return 0.0

    compactness = (volume_mm3 ** (2/3)) / surface_area_mm2
    return float(compactness)


def _compute_intensity_features(
    image: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute intensity-based features within the masked region.

    Args:
        image: Original image intensity values.
        mask: Binary mask defining the region of interest.

    Returns:
        Dictionary of intensity features.
    """
    masked_values = image[mask > 0]

    if len(masked_values) == 0:
        return {
            "Intensity_Mean": 0.0,
            "Intensity_Std": 0.0,
            "Intensity_Min": 0.0,
            "Intensity_Max": 0.0,
            "Intensity_Median": 0.0,
        }

    return {
        "Intensity_Mean": float(np.mean(masked_values)),
        "Intensity_Std": float(np.std(masked_values)),
        "Intensity_Min": float(np.min(masked_values)),
        "Intensity_Max": float(np.max(masked_values)),
        "Intensity_Median": float(np.median(masked_values)),
        "Intensity_Range": float(np.ptp(masked_values)),  # Peak-to-peak (max - min)
    }


def _get_default_features() -> dict[str, float]:
    """Return default feature values when extraction fails."""
    return {
        "Volume_mm3": 0.0,
        "Surface_Area_mm2": 0.0,
        "Surface_Volume_Ratio": 0.0,
        "Sphericity": 0.0,
        "Elongation": 1.0,
        "Compactness": 0.0,
        "Aspect_Ratio": 0.0,
        "SR_Depth_to_Mean": 0.0,
        "SR_Max_to_Mean": 0.0,
        "Max_Diameter_mm": 0.0,
        "Min_Diameter_mm": 0.0,
        "Mean_Diameter_mm": 0.0,
        "BoundingBox_Length_mm": 0.0,
        "BoundingBox_Width_mm": 0.0,
        "BoundingBox_Depth_mm": 0.0,
        "Intensity_Mean": 0.0,
        "Intensity_Std": 0.0,
        "Intensity_Min": 0.0,
        "Intensity_Max": 0.0,
        "Intensity_Median": 0.0,
        "Intensity_Range": 0.0,
    }
