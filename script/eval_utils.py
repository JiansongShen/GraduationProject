from __future__ import annotations

"""Reusable helpers for segmentation evaluation and inference.

The goal of this module is to keep `script/eval_split.py`, `script/train_split.py`,
and the UI inference path focused on orchestration instead of patch stitching or
loss definitions.
"""

import logging
from pathlib import Path

import SimpleITK as sitk
import torch

from data.MedicalPatchDataset import MedicalPatchDataset
from data.data_preprocesser import resample_in_memory


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice loss on probability tensors in `[0, 1]`."""
    pred_prob = torch.sigmoid(pred) if pred.dtype.is_floating_point else pred.float()
    pred_flat = pred_prob.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1.0 - dice_coeff


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy + Dice loss for segmentation."""
    pred_prob = torch.sigmoid(pred)
    bce = torch.nn.functional.binary_cross_entropy(pred_prob, target)
    dice = dice_loss(pred_prob, target)
    return (bce + dice) / 2


def align_target_shape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Make the target shape compatible with a 5D segmentation prediction."""
    if target.ndim == pred.ndim - 1:
        target = target.unsqueeze(1)
    if target.shape != pred.shape:
        raise ValueError(
            f"Prediction/target shape mismatch after alignment: pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    return torch.clamp(target.float(), 0.0, 1.0)


def sequential_patch_prediction(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    *,
    case_index: int,
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.Tensor], tuple[int, int, int], Path, tuple[float, float, float], tuple[float, ...], tuple[float, float, float]]:
    """Run deterministic patch inference for one case and return probability patches.

    This helper is used by training-time visualization/export so we can save one
    stitched prediction per epoch without disturbing the training sampling policy.
    """
    case = dataset.cases[case_index]
    original_image = sitk.ReadImage(case.image_path)
    original_spacing = original_image.GetSpacing()
    original_direction = original_image.GetDirection()
    original_origin = original_image.GetOrigin()

    image_tensor, _ = dataset.get_src_item(case_index)
    images, _labels = dataset.get_patches(case_index, sampling_mode="sequential")
    patch_count = int(images.shape[0])
    if patch_count == 0:
        raise ValueError(f"No sequential patches available for case: {case.image_path}")

    probability_patches: list[torch.Tensor] = []
    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        for patch_start in range(0, patch_count, batch_size):
            patch_end = min(patch_start + batch_size, patch_count)
            batch_images = images[patch_start:patch_end].to(device)
            outputs = model(batch_images)
            probabilities = torch.sigmoid(outputs).detach().cpu()
            for output_tensor in probabilities:
                probability_patches.append(output_tensor)
    if model_was_training:
        model.train()

    return (
        probability_patches,
        tuple(int(dim) for dim in image_tensor.shape),
        Path(case.image_path),
        original_spacing,
        original_direction,
        original_origin,
    )


def combine_to_nifti(
    unified_spacing_patch_list: list[torch.Tensor],
    patch_shape: tuple[int, int, int],
    src_shape: tuple[int, int, int],
    *,
    binarize: bool = False,
    nifti_path: Path,
) -> sitk.Image:
    """Stitch sequential patch predictions back into a SimpleITK volume."""
    if not unified_spacing_patch_list:
        raise ValueError("Cannot stitch eval prediction: unified_spacing_patch_list is empty.")

    patch_d, patch_h, patch_w = patch_shape
    src_d, src_h, src_w = src_shape
    combined = torch.zeros(src_shape, dtype=torch.uint8 if binarize else torch.float32)
    patch_idx = 0
    prob_min = float("inf")
    prob_max = float("-inf")
    prob_sum = 0.0
    prob_count = 0
    foreground_voxels = 0

    src_img = sitk.ReadImage(str(nifti_path))
    resampled_img = resample_in_memory(src_img)
    resampled_img_d, resampled_img_h, resampled_img_w = resampled_img.GetSize()
    expected_patch_count = (
        ((resampled_img_d + patch_d - 1) // patch_d)
        * ((resampled_img_h + patch_h - 1) // patch_h)
        * ((resampled_img_w + patch_w - 1) // patch_w)
    )

    logging.info(
        "Combining patches into nifti: patch_shape=%s src_shape=%s expected_patches=%s provided_patches=%s binarize=%s",
        patch_shape,
        src_shape,
        expected_patch_count,
        len(unified_spacing_patch_list),
        binarize,
    )
    if expected_patch_count != len(unified_spacing_patch_list):
        raise ValueError(f"Invalid patch list size: {len(unified_spacing_patch_list)}, expected {expected_patch_count}")

    for z in range(0, src_d, patch_d):
        for y in range(0, src_h, patch_h):
            for x in range(0, src_w, patch_w):
                if patch_idx >= len(unified_spacing_patch_list):
                    image = sitk.GetImageFromArray(combined.numpy())
                    return sitk.Cast(image, sitk.sitkUInt8) if binarize else image

                patch = unified_spacing_patch_list[patch_idx].detach().cpu()
                while patch.ndim > 3:
                    patch = patch[0]
                if tuple(patch.shape) != patch_shape:
                    raise ValueError(f"Invalid eval patch shape: {tuple(patch.shape)}, expected {patch_shape}")

                z_end = min(z + patch_d, src_d)
                y_end = min(y + patch_h, src_h)
                x_end = min(x + patch_w, src_w)
                actual_patch = patch[: z_end - z, : y_end - y, : x_end - x].float()
                prob_patch = torch.sigmoid(actual_patch)
                # Ensure probability values are in [0, 1]
                # prob_patch = torch.clamp(prob_patch, 0.0, 1.0)
                prob_min = min(prob_min, float(prob_patch.min().item()))
                prob_max = max(prob_max, float(prob_patch.max().item()))
                prob_sum += float(prob_patch.sum().item())
                prob_count += int(prob_patch.numel())

                if binarize:
                    bin_patch = (prob_patch > 0.5).to(torch.uint8)
                    foreground_voxels += int(bin_patch.sum().item())
                    combined[z:z_end, y:y_end, x:x_end] = bin_patch
                else:
                    combined[z:z_end, y:y_end, x:x_end] = prob_patch
                patch_idx += 1

    logging.info(
        "Finished combining patches: patches_used=%s prob_min=%.6f prob_max=%.6f prob_mean=%.6f foreground_voxels=%s",
        patch_idx,
        prob_min if prob_count else float("nan"),
        prob_max if prob_count else float("nan"),
        (prob_sum / prob_count) if prob_count else float("nan"),
        foreground_voxels,
    )
    image = sitk.GetImageFromArray(combined.numpy())
    return sitk.Cast(image, sitk.sitkUInt8) if binarize else image
