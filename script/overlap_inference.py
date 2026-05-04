"""Overlapping patch inference for 3D medical image segmentation.

This module implements overlapping patch-based prediction to significantly reduce
boundary artifacts and prediction discontinuities at patch edges, which are common
issues in sliding-window based segmentation approaches.

Key concepts:
- Patch size: Full size of each extracted patch (e.g., 64×64×64)
- Effective size: Center region kept for final output (e.g., 48×48×48)
- Stride: Step between adjacent patches (equals effective_size for seamless coverage)
- Gaussian blending: Center voxels receive higher weights than edge voxels

Example configuration (recommended for 64×64×64 patches):
    - patch_size: [64, 64, 64]
    - effective_size: [48, 48, 48]
    - stride: 48 (equals effective_size)
    - overlap: 16 voxels on each side
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger("gradulate.overlap_inference")


@dataclass
class OverlapInferenceConfig:
    """Configuration for overlapping patch inference.

    Attributes:
        enabled: Enable overlapping inference (if False, uses standard non-overlapping inference)
        patch_size: Full patch dimensions (D, H, W) to extract from the volume
        effective_size: Center region (D, H, W) to keep from each patch prediction.
                       This should be smaller than patch_size to exclude boundary artifacts.
                       Stride will automatically be set to effective_size.
        padding_mode: How to handle volume boundaries:
            - "reflect": Mirror padding (recommended, avoids edge artifacts)
            - "constant": Pad with padding_value
            - "none": No padding (may leave gaps at boundaries)
        padding_value: Fill value when padding_mode is "constant"
        blend_mode: How to combine overlapping predictions:
            - "gaussian": Gaussian-weighted blending (recommended, smooth transitions)
            - "average": Simple average of overlapping regions
        gaussian_sigma: Standard deviation for Gaussian weighting, relative to effective_size.
                       Higher values = more gradual blending. Typical range: 0.3-0.5
        use_amp: Enable automatic mixed precision (FP16) for memory efficiency
        batch_size: Number of patches to process simultaneously
    """
    enabled: bool = True
    patch_size: tuple[int, int, int] = (64, 64, 64)
    effective_size: tuple[int, int, int] = (48, 48, 48)
    padding_mode: str = "reflect"
    padding_value: float = 0.0
    blend_mode: str = "gaussian"
    gaussian_sigma: float = 0.4
    use_amp: bool = True
    batch_size: int = 4

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.enabled:
            # Validate sizes
            for i, (ps, es) in enumerate(zip(self.patch_size, self.effective_size)):
                if ps <= 0 or es <= 0:
                    raise ValueError(f"Dimension {i}: sizes must be positive")
                if es > ps:
                    raise ValueError(
                        f"effective_size[{i}]={es} cannot exceed patch_size[{i}]={ps}"
                    )

            # Validate blend_mode
            if self.blend_mode not in ("gaussian", "average"):
                raise ValueError(
                    f"blend_mode must be 'gaussian' or 'average', got '{self.blend_mode}'"
                )

            # Validate padding_mode
            if self.padding_mode not in ("reflect", "constant", "none"):
                raise ValueError(
                    f"padding_mode must be 'reflect', 'constant', or 'none', "
                    f"got '{self.padding_mode}'"
                )

            # Validate gaussian_sigma
            if self.gaussian_sigma <= 0:
                raise ValueError(f"gaussian_sigma must be positive, got {self.gaussian_sigma}")

            # Validate batch_size
            if self.batch_size <= 0:
                raise ValueError(f"batch_size must be positive, got {self.batch_size}")

            # Log recommended stride
            logger.debug(
                "Overlap inference config: patch=%s, effective=%s, stride=%s, "
                "overlap=%s, blend=%s",
                self.patch_size,
                self.effective_size,
                self.effective_size,  # stride = effective_size
                tuple(p - e for p, e in zip(self.patch_size, self.effective_size)),
                self.blend_mode,
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OverlapInferenceConfig":
        """Create config from dictionary with automatic type conversion.

        Args:
            data: Dictionary with configuration parameters. List values for
                  patch_size/effective_size are automatically converted to tuples.

        Returns:
            OverlapInferenceConfig instance (disabled if data is empty or None)
        """
        if not data:
            return cls(enabled=False)

        kwargs = dict(data)
        # Convert list to tuple for dimension parameters
        if "patch_size" in kwargs and isinstance(kwargs["patch_size"], list):
            kwargs["patch_size"] = tuple(kwargs["patch_size"])
        if "effective_size" in kwargs and isinstance(kwargs["effective_size"], list):
            kwargs["effective_size"] = tuple(kwargs["effective_size"])

        return cls(**kwargs)


def create_gaussian_weight_map(
    effective_size: tuple[int, int, int],
    sigma: float = 0.4,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a 3D Gaussian weight map for blending overlapping predictions.

    The Gaussian kernel is centered at the middle of the volume with sigma
    specified as a fraction of the size (e.g., sigma=0.4 means sigma = 0.4 * size).

    Higher weights at the center encourage smoother blending where predictions
    overlap, reducing visible seams between patches.

    Args:
        effective_size: Center region dimensions (D, H, W)
        sigma: Standard deviation as fraction of size (0.3-0.5 recommended)
        device: Target device for the weight tensor

    Returns:
        Weight tensor of shape [effective_size], values in (0, 1]
    """
    d, h, w = effective_size

    # Create coordinate grids normalized to [-1, 1] range
    z_coords = torch.linspace(-1, 1, d, device=device)
    y_coords = torch.linspace(-1, 1, h, device=device)
    x_coords = torch.linspace(-1, 1, w, device=device)

    zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

    # Calculate sigma in absolute coordinates (fraction of [-1, 1] range)
    sigma_abs = sigma  # sigma is already relative since coords are in [-1, 1]

    # Compute 3D Gaussian
    squared_distance = zz**2 + yy**2 + xx**2
    weights = torch.exp(-squared_distance / (2 * sigma_abs**2))

    # Normalize so maximum weight is 1
    weights = weights / weights.max()

    return weights


def generate_overlap_patch_positions(
    volume_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    effective_size: tuple[int, int, int],
    stride: tuple[int, int, int] | None = None,
    padding_mode: str = "none",
    padding_value: float = 0.0,
) -> tuple[list[tuple[int, int, int]], torch.Tensor]:
    """Generate starting positions for overlapping patches covering a volume.

    This function calculates where to place patches so that every voxel in the
    volume (or padded volume) is covered by at least one patch's effective region.

    Args:
        volume_shape: Original volume dimensions (D, H, W)
        patch_size: Full patch size (D, H, W)
        effective_size: Effective center region kept from each patch (D, H, W)
        stride: Step between patch centers. If None, defaults to effective_size.
                Smaller strides increase overlap and memory usage.
        padding_mode: "reflect", "constant", or "none"
        padding_value: Value for constant padding

    Returns:
        Tuple of:
            - List of (z, y, x) starting positions for each patch
            - Padded volume tensor ready for patch extraction
    """
    if stride is None:
        stride = effective_size

    # Calculate padding needed
    overlap = tuple(p - e for p, e in zip(patch_size, effective_size))
    pad_before = tuple(o // 2 for o in overlap)
    pad_after = tuple(o - pb for o, pb in zip(overlap, pad_before))

    # Pad volume
    d, h, w = volume_shape
    d_pad = pad_before[0] + pad_after[0]
    h_pad = pad_before[1] + pad_after[1]
    w_pad = pad_before[2] + pad_after[2]

    # Create padded volume (this would be done on the actual volume in predict_with_overlap)
    padded_shape = (
        d + pad_before[0] + pad_after[0],
        h + pad_before[1] + pad_after[1],
        w + pad_before[2] + pad_after[2],
    )

    # Generate patch positions in padded coordinate space
    positions: list[tuple[int, int, int]] = []

    for z in range(-pad_before[0], d + pad_before[0], stride[0]):
        for y in range(-pad_before[1], h + pad_before[1], stride[1]):
            for x in range(-pad_before[2], w + pad_before[2], stride[2]):
                # Clip to valid range within padded volume
                z_start = max(0, min(z, padded_shape[0] - patch_size[0]))
                y_start = max(0, min(y, padded_shape[1] - patch_size[1]))
                x_start = max(0, min(x, padded_shape[2] - patch_size[2]))
                positions.append((z_start, y_start, x_start))

    # Remove duplicates while preserving order
    seen = set()
    unique_positions = []
    for pos in positions:
        if pos not in seen:
            seen.add(pos)
            unique_positions.append(pos)

    return unique_positions, torch.zeros(padded_shape)  # placeholder for padding


def _extract_patch(
    volume: torch.Tensor,
    start_pos: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Extract a patch from a 3D volume.

    Args:
        volume: Volume tensor of shape [C, D, H, W] or [D, H, W]
        start_pos: Starting (z, y, x) position
        patch_size: Size of patch to extract

    Returns:
        Extracted patch tensor
    """
    z, y, x = start_pos
    d, h, w = patch_size

    if volume.ndim == 3:
        return volume[z : z + d, y : y + h, x : x + w]
    elif volume.ndim == 4:
        return volume[:, z : z + d, y : y + h, x : x + w]
    else:
        raise ValueError(f"Expected 3D or 4D volume, got {volume.ndim}D")


def _apply_padding(
    volume: torch.Tensor,
    pad_before: tuple[int, int, int],
    pad_after: tuple[int, int, int],
    mode: str = "reflect",
    value: float = 0.0,
) -> torch.Tensor:
    """Apply padding to a 3D or 4D volume tensor.

    Args:
        volume: Input tensor [C, D, H, W] or [D, H, W]
        pad_before: Padding before each dimension
        pad_after: Padding after each dimension
        mode: "reflect", "constant", or "replicate"
        value: Fill value for constant padding

    Returns:
        Padded tensor
    """
    if mode == "none":
        return volume

    if volume.ndim == 3:
        pad_dims = [pad_before[2], pad_after[2], pad_before[1], pad_after[1], pad_before[0], pad_after[0]]
    elif volume.ndim == 4:
        pad_dims = [pad_before[2], pad_after[2], pad_before[1], pad_after[1], pad_before[0], pad_after[0]]
    else:
        raise ValueError(f"Expected 3D or 4D volume, got {volume.ndim}D")

    if mode == "constant":
        return F.pad(volume, pad_dims, mode=mode, value=value)
    elif mode in ("reflect", "replicate"):
        return F.pad(volume, pad_dims, mode=mode)
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")


def predict_with_overlap(
    model: torch.nn.Module,
    image: torch.Tensor,
    config: OverlapInferenceConfig,
    device: torch.device,
    show_progress: bool = False,
) -> torch.Tensor:
    """Run overlapping patch inference on a 3D medical image.

    This function implements overlapping sliding-window inference to reduce boundary
    artifacts. Each patch extracts a larger region than it outputs, and only the
    center (effective) region is used for the final stitched result.

    Process:
    1. Pad the input volume to handle boundaries
    2. Extract patches at regular strides
    3. Run model inference on each patch (in batches)
    4. Extract effective center region from each prediction
    5. Stitch with Gaussian-weighted blending
    6. Return prediction in original volume coordinates

    Args:
        model: Segmentation model (should output probabilities in [0, 1])
        image: Input image tensor of shape [1, 1, D, H, W] or [D, H, W]
        config: Overlapping inference configuration
        device: Compute device
        show_progress: Show progress bar during inference

    Returns:
        Prediction tensor of shape [1, 1, D, H, W] (same spatial shape as input)
    """
    if not config.enabled:
        logger.warning("Overlap inference called but config.enabled=False, falling back to standard inference")
        return _standard_inference(model, image, device)

    model.eval()

    # Capture original input shape for return format determination
    original_ndim = image.ndim
    original_shape = image.shape if image.ndim >= 3 else None

    # Ensure 5D input: [B, C, D, H, W]
    input_5d = image
    if image.ndim == 3:
        input_5d = image.unsqueeze(0).unsqueeze(0)  # [D, H, W] -> [1, 1, D, H, W]
    elif image.ndim == 4:
        input_5d = image.unsqueeze(0)  # [1, D, H, W] -> [1, 1, D, H, W]

    # Extract shape info
    batch_size, channels, d, h, w = input_5d.shape
    volume_shape = (d, h, w)
    patch_size = config.patch_size
    effective_size = config.effective_size

    # Calculate overlap and offsets
    overlap = tuple(p - e for p, e in zip(patch_size, effective_size))
    offset = tuple(o // 2 for o in overlap)

    # Pad volume
    pad_before = (offset[0], offset[1], offset[2])
    pad_after = (offset[0], offset[1], offset[2])
    padded_volume = _apply_padding(
        input_5d[0],  # [C, D, H, W]
        pad_before,
        pad_after,
        mode=config.padding_mode,
        value=config.padding_value,
    )  # [C, D', H', W']

    padded_shape = tuple(padded_volume.shape)

    # Calculate stride (defaults to effective_size for seamless coverage)
    stride = effective_size

    # Generate patch positions in padded coordinates
    patch_starts: list[tuple[int, int, int]] = []
    for z in range(0, padded_shape[1] - patch_size[0] + 1, stride[0]):
        for y in range(0, padded_shape[2] - patch_size[1] + 1, stride[1]):
            for x in range(0, padded_shape[3] - patch_size[2] + 1, stride[2]):
                patch_starts.append((z, y, x))

    num_patches = len(patch_starts)
    logger.info(
        "Overlap inference: volume=%s, padded=%s, patches=%d, "
        "patch=%s, effective=%s, stride=%s",
        volume_shape,
        padded_shape,
        num_patches,
        patch_size,
        effective_size,
        stride,
    )

    # Create weight map for blending (keep on CPU since we accumulate to CPU tensors)
    if config.blend_mode == "gaussian":
        weight_map = create_gaussian_weight_map(
            effective_size,
            sigma=config.gaussian_sigma,
            device="cpu",
        )
    else:  # average
        weight_map = torch.ones(effective_size, device="cpu", dtype=torch.float32)

    # Initialize accumulators
    prediction_sum = torch.zeros(volume_shape, dtype=torch.float32, device="cpu")
    weight_sum = torch.zeros(volume_shape, dtype=torch.float32, device="cpu")

    # Process patches in batches
    batch_size = config.batch_size
    use_amp = config.use_amp and device.type == "cuda"

    if show_progress:
        patch_iter = tqdm(patch_starts, desc="Overlapping inference")
    else:
        patch_iter = patch_starts

    with torch.no_grad():
        for batch_start in range(0, num_patches, batch_size):
            batch_end = min(batch_start + batch_size, num_patches)
            batch_positions = patch_starts[batch_start:batch_end]

            # Extract batch of patches
            batch_patches = []
            for pos in batch_positions:
                patch = _extract_patch(padded_volume, pos, patch_size)
                batch_patches.append(patch)

            batch_tensor = torch.stack(batch_patches, dim=0).to(device)  # [B, C, D, H, W]

            # Run inference
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(batch_tensor)
            else:
                outputs = model(batch_tensor)

            # Move to CPU for accumulation
            outputs_cpu = outputs.detach().cpu().float()

            # Accumulate predictions
            for i, pos in enumerate(batch_positions):
                output = outputs_cpu[i]  # [C, D, H, W] or [D, H, W] after batch removal

                # Handle various model output formats:
                # [B, C, D, H, W] -> batch[i] -> [C, D, H, W]
                # [B, D, H, W] -> batch[i] -> [D, H, W]
                # Where C=1 for single-channel segmentation output

                # If output has 4 dimensions, it likely includes the channel dimension
                if output.ndim == 4:
                    # Check if first dim is the channel (should be 1 for single-channel)
                    if output.shape[0] == 1:
                        output = output[0]  # Remove channel: [1, D, H, W] -> [D, H, W]
                    # else: [D, H, W] format, no channel dim

                # Now output should be [D, H, W]

                # Extract effective center region
                z, y, x = pos
                effective_pred = output[
                    offset[0] : offset[0] + effective_size[0],
                    offset[1] : offset[1] + effective_size[1],
                    offset[2] : offset[2] + effective_size[2],
                ]

                # Map back to original volume coordinates
                z_orig = z - offset[0]
                y_orig = y - offset[1]
                x_orig = x - offset[2]

                # Clip to valid range
                z_end = min(z_orig + effective_size[0], volume_shape[0])
                y_end = min(y_orig + effective_size[1], volume_shape[1])
                x_end = min(x_orig + effective_size[2], volume_shape[2])

                z_start_clip = max(0, -z_orig)
                y_start_clip = max(0, -y_orig)
                x_start_clip = max(0, -x_orig)

                z_orig = max(0, z_orig)
                y_orig = max(0, y_orig)
                x_orig = max(0, x_orig)

                # Accumulate weighted prediction
                prediction_sum[
                    z_orig:z_orig + (z_end - z_orig),
                    y_orig:y_orig + (y_end - y_orig),
                    x_orig:x_orig + (x_end - x_orig),
                ] += (
                    effective_pred[
                        z_start_clip : z_start_clip + (z_end - z_orig),
                        y_start_clip : y_start_clip + (y_end - y_orig),
                        x_start_clip : x_start_clip + (x_end - x_orig),
                    ]
                    * weight_map[
                        z_start_clip : z_start_clip + (z_end - z_orig),
                        y_start_clip : y_start_clip + (y_end - y_orig),
                        x_start_clip : x_start_clip + (x_end - x_orig),
                    ]
                )

                weight_sum[
                    z_orig:z_orig + (z_end - z_orig),
                    y_orig:y_orig + (y_end - y_orig),
                    x_orig:x_orig + (x_end - x_orig),
                ] += weight_map[
                    z_start_clip : z_start_clip + (z_end - z_orig),
                    y_start_clip : y_start_clip + (y_end - y_orig),
                    x_start_clip : x_start_clip + (x_end - x_orig),
                ]

    # Normalize by weights
    weight_sum = weight_sum.clamp(min=1e-8)
    prediction = prediction_sum / weight_sum

    # Clamp to valid probability range
    prediction = prediction.clamp(0.0, 1.0)

    # Return in same format as input
    if original_ndim == 3:
        return prediction  # [D, H, W]
    elif original_ndim == 4:
        return prediction.unsqueeze(0)  # [1, D, H, W]
    else:
        return prediction.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]


def _standard_inference(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Standard non-overlapping patch inference (fallback when overlap disabled).

    Args:
        model: Segmentation model
        image: Input tensor
        device: Compute device

    Returns:
        Prediction tensor
    """
    with torch.no_grad():
        if image.ndim == 3:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.ndim == 4:
            image = image.unsqueeze(0)

        output = model(image.to(device))
        return output.squeeze(0).cpu()


# Convenience alias for import compatibility
OverlapInferenceConfigProtocol = OverlapInferenceConfig
