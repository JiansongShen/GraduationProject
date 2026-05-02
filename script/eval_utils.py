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
    pred_prob = pred.float()
    pred_flat = pred_prob.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1.0 - dice_coeff


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Stable segmentation loss built from logits.

    We use BCEWithLogitsLoss so the model can emit raw logits without an extra
    sigmoid in the forward pass. Dice still operates on probabilities derived
    from those logits.
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    pred_prob = torch.sigmoid(pred)
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
    """Run deterministic patch inference for one case and return probability patches."""
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
            logits = model(batch_images)
            logging.debug(
                "seq-predict batch=%s:%s logits shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                patch_start,
                patch_end,
                tuple(logits.shape),
                logits.dtype,
                float(logits.min().item()),
                float(logits.max().item()),
                float(logits.mean().item()),
            )
            probabilities = logits.detach().cpu()
            logging.debug(
                "seq-predict batch=%s:%s prob shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f",
                patch_start,
                patch_end,
                tuple(probabilities.shape),
                probabilities.dtype,
                float(probabilities.min().item()),
                float(probabilities.max().item()),
                float(probabilities.mean().item()),
            )
            for output_tensor in probabilities:
                probability_patches.append(output_tensor)
    if model_was_training:
        model.train()

    shape: tuple[int, int, int] = image_tensor.shape

    return (
        probability_patches,
        shape,
        Path(case.image_path),
        original_spacing,
        original_direction,
        original_origin,
    )


def stitch_probability_patches(
    patch_list: list[torch.Tensor],
    patch_shape: tuple[int, int, int],
    src_shape: tuple[int, int, int],
    *,
    nifti_path: Path,
    binarize: bool = False,
) -> tuple[sitk.Image, dict[str, float]]:
    """Stitch sequential patch predictions into the resampled grid."""
    if not patch_list:
        raise ValueError("Cannot stitch eval prediction: patch_list is empty.")

    patch_d, patch_h, patch_w = patch_shape
    src_img = sitk.ReadImage(str(nifti_path))
    resampled_img = resample_in_memory(src_img)
    resampled_shape = tuple(int(dim) for dim in sitk.GetArrayFromImage(resampled_img).shape)
    if resampled_shape != src_shape:
        raise ValueError(f"shape mismatch: expected src_shape={src_shape}, got resampled_shape={resampled_shape}")

    combined = torch.zeros(resampled_shape, dtype=torch.uint8 if binarize else torch.float32)
    expected_patch_count = (
        ((resampled_shape[0] + patch_d - 1) // patch_d)
        * ((resampled_shape[1] + patch_h - 1) // patch_h)
        * ((resampled_shape[2] + patch_w - 1) // patch_w)
    )

    logging.info(
        "Combining patches into nifti: patch_shape=%s resampled_shape=%s expected_patches=%s provided_patches=%s binarize=%s",
        patch_shape,
        resampled_shape,
        expected_patch_count,
        len(patch_list),
        binarize,
    )
    if expected_patch_count != len(patch_list):
        raise ValueError(f"Invalid patch list size: {len(patch_list)}, expected {expected_patch_count}")

    patch_idx = 0
    prob_min = float("inf")
    prob_max = float("-inf")
    prob_sum = 0.0
    prob_count = 0
    foreground_voxels = 0

    for z in range(0, resampled_shape[0], patch_d):
        for y in range(0, resampled_shape[1], patch_h):
            for x in range(0, resampled_shape[2], patch_w):
                patch = patch_list[patch_idx].detach().cpu()
                while patch.ndim > 3:
                    patch = patch[0]
                if tuple(patch.shape) != patch_shape:
                    raise ValueError(f"Invalid eval patch shape: {tuple(patch.shape)}, expected {patch_shape}")

                z_end = min(z + patch_d, resampled_shape[0])
                y_end = min(y + patch_h, resampled_shape[1])
                x_end = min(x + patch_w, resampled_shape[2])
                actual_patch = patch[: z_end - z, : y_end - y, : x_end - x].float()
                prob_patch = actual_patch.clamp(0.0, 1.0)

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

    stats = {
        "patches_used": float(patch_idx),
        "prob_min": prob_min if prob_count else float("nan"),
        "prob_max": prob_max if prob_count else float("nan"),
        "prob_mean": (prob_sum / prob_count) if prob_count else float("nan"),
        "foreground_voxels": float(foreground_voxels),
    }
    logging.debug(
        "Finished combining patches: patches_used=%s prob_min=%.6f prob_max=%.6f prob_mean=%.6f foreground_voxels=%s",
        patch_idx,
        stats["prob_min"],
        stats["prob_max"],
        stats["prob_mean"],
        foreground_voxels,
    )

    resampled_prediction = sitk.GetImageFromArray(combined.numpy())
    resampled_prediction.CopyInformation(resampled_img)
    logging.debug(
        "resampled prediction geometry: size=%s spacing=%s origin=%s direction_len=%s",
        resampled_prediction.GetSize(),
        resampled_prediction.GetSpacing(),
        resampled_prediction.GetOrigin(),
        len(resampled_prediction.GetDirection()),
    )
    return resampled_prediction, stats


def restore_prediction_to_source_grid(
    prediction_image: sitk.Image,
    source_image: sitk.Image,
    *,
    binarize: bool = False,
) -> sitk.Image:
    """Map a prediction from the resampled grid back to the original image grid."""
    interpolator = sitk.sitkNearestNeighbor if binarize else sitk.sitkLinear
    logging.debug(
        "restoring prediction to source grid: pred_size=%s pred_spacing=%s src_size=%s src_spacing=%s",
        prediction_image.GetSize(),
        prediction_image.GetSpacing(),
        source_image.GetSize(),
        source_image.GetSpacing(),
    )
    restored_prediction = sitk.Resample(
        prediction_image,
        source_image,
        sitk.Transform(),
        interpolator,
        0.0,
        sitk.sitkUInt8 if binarize else sitk.sitkFloat32,
    )
    logging.debug(
        "restored prediction geometry: size=%s spacing=%s origin=%s direction_len=%s",
        restored_prediction.GetSize(),
        restored_prediction.GetSpacing(),
        restored_prediction.GetOrigin(),
        len(restored_prediction.GetDirection()),
    )
    if restored_prediction.GetSize() != source_image.GetSize():
        raise ValueError("not successfully combine")
    return sitk.Cast(restored_prediction, sitk.sitkUInt8) if binarize else restored_prediction


def combine_to_nifti(
    unified_spacing_patch_list: list[torch.Tensor],
    patch_shape: tuple[int, int, int],
    src_shape: tuple[int, int, int],
    *,
    binarize: bool = False,
    nifti_path: Path,
) -> sitk.Image:
    """Convenience wrapper that stitches patches and restores original geometry."""
    source_image = sitk.ReadImage(str(nifti_path))
    resampled_prediction, _stats = stitch_probability_patches(
        unified_spacing_patch_list,
        patch_shape,
        src_shape,
        nifti_path=nifti_path,
        binarize=binarize,
    )
    return restore_prediction_to_source_grid(resampled_prediction, source_image, binarize=binarize)
