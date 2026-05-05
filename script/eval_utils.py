from __future__ import annotations

"""Reusable helpers for segmentation evaluation and inference.

The goal of this module is to keep `script/eval_split.py`, `script/train_split.py`,
and the UI inference path focused on orchestration instead of patch stitching or
loss definitions.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import SimpleITK as sitk
import torch
import torch.nn.functional as F

from data.MedicalPatchDataset import MedicalPatchDataset
from data.data_preprocesser import resample_in_memory

if TYPE_CHECKING:
    from script.overlap_inference import OverlapInferenceConfig


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice loss on probability tensors in `[0, 1]`."""
    pred_prob = pred.float()
    pred_flat = pred_prob.reshape(-1)
    target_flat = target.reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1.0 - dice_coeff


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Soft Dice coefficient in ``[0, 1]`` (same smoothing as ``dice_loss``)."""
    return 1.0 - dice_loss(pred, target, smooth=smooth)


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Tversky loss = ``1 - TI`` on soft masks; ``beta > alpha`` penalizes false negatives more."""
    p = pred.float().reshape(-1)
    t = target.float().reshape(-1)
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    fn = ((1.0 - p) * t).sum()
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky_index


def focal_dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 4.0 / 3.0,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Focal Dice: ``(1 - DSC) ** gamma`` with soft Dice coefficient ``DSC``."""
    dsc = dice_coefficient(pred, target, smooth=smooth)
    return (1.0 - dsc).clamp(min=0.0).pow(gamma)


def _surface_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    *,
    surface: str,
    tversky_alpha: float,
    tversky_beta: float,
    focal_dice_gamma: float,
    smooth: float,
) -> torch.Tensor:
    surface = surface.lower()
    if surface == "dice":
        return dice_loss(pred_prob, target, smooth=smooth)
    if surface == "tversky":
        return tversky_loss(pred_prob, target, alpha=tversky_alpha, beta=tversky_beta, smooth=smooth)
    if surface == "focal_dice":
        return focal_dice_loss(pred_prob, target, gamma=focal_dice_gamma, smooth=smooth)
    raise ValueError(f"Unknown segmentation_surface_loss: {surface!r} (use dice, tversky, focal_dice)")


def combined_loss_with_parts(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    *,
    surface: str = "tversky",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_dice_gamma: float = 4.0 / 3.0,
    smooth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(combined_loss, bce, surface_loss)``.

    The third value is the chosen surface term (Dice, Tversky, or Focal Dice), not always
    classical Dice loss. For validation **metrics**, compute standard Dice with ``dice_loss``
    or ``dice_coefficient`` explicitly.
    """
    bce = F.binary_cross_entropy(pred_prob, target)
    surface_term = _surface_loss(
        pred_prob,
        target,
        surface=surface,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        focal_dice_gamma=focal_dice_gamma,
        smooth=smooth,
    )
    total = (bce)
    return total, bce, surface_term


def combined_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    *,
    surface: str = "tversky",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_dice_gamma: float = 4.0 / 3.0,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Mean of voxel-wise BCE and the configured surface loss on probability maps.

    ``AttentionUnet`` ends with ``Sigmoid`` (``UnetDecoder``), so ``pred_prob``
    is already in ``(0, 1)``. No extra ``sigmoid`` here — that would mean
    treating probabilities as logits or applying sigmoid twice to logits.
    """
    total, _, _ = combined_loss_with_parts(
        pred_prob,
        target,
        surface=surface,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        focal_dice_gamma=focal_dice_gamma,
        smooth=smooth,
    )
    return total


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

    with torch.no_grad():
        for patch_start in range(0, patch_count, batch_size):
            patch_end = min(patch_start + batch_size, patch_count)
            batch_images = images[patch_start:patch_end].to(device)
            prob_map = model(batch_images).detach().cpu()
            for output_tensor in prob_map:
                while output_tensor.ndim > 4:
                    output_tensor = output_tensor[0]
                if output_tensor.ndim == 4:
                    output_tensor = output_tensor[0]  # Remove channel dim
                probability_patches.append(output_tensor)

    shape: tuple[int, int, int] = tuple(int(d) for d in image_tensor.shape)

    return (
        probability_patches,
        shape,
        Path(case.image_path),
        original_spacing,
        original_direction,
        original_origin,
    )


def overlap_patch_prediction(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    *,
    case_index: int,
    device: torch.device,
    overlap_config: "OverlapInferenceConfig",
) -> tuple[torch.Tensor, tuple[int, int, int], Path, tuple[float, float, float], tuple[float, ...], tuple[float, float, float], dict]:
    """Run overlapping patch inference for one case with Gaussian-weighted blending.

    Args:
        model: 分割模型
        dataset: 医疗图像数据集
        case_index: 病例索引
        device: 计算设备
        overlap_config: 重叠推理配置

    Returns:
        (prediction, source_shape, image_path, original_spacing, original_direction, original_origin, stats)
    """
    from script.overlap_inference import predict_with_overlap as _predict_overlap

    case = dataset.cases[case_index]
    original_image = sitk.ReadImage(case.image_path)
    original_spacing = original_image.GetSpacing()
    original_direction = original_image.GetDirection()
    original_origin = original_image.GetOrigin()

    image_tensor, _ = dataset.get_src_item(case_index)
    source_shape = tuple(int(d) for d in image_tensor.shape)

    # 转换为 [1, 1, D, H, W] 格式
    input_tensor = image_tensor.unsqueeze(0).unsqueeze(0).float()

    model.eval()
    with torch.no_grad():
        prediction = _predict_overlap(
            model=model,
            image=input_tensor,
            config=overlap_config,
            device=device,
        )

    # 收集统计信息
    pred_np = prediction.squeeze().cpu().numpy()
    stats = {
        "prob_min": float(pred_np.min()),
        "prob_max": float(pred_np.max()),
        "prob_mean": float(pred_np.mean()),
        "prob_std": float(pred_np.std()),
        "inference_mode": "overlap",
    }

    return (
        prediction.squeeze(0),  # [1, D, H, W] -> [D, H, W]
        source_shape,
        Path(case.image_path),
        original_spacing,
        original_direction,
        original_origin,
        stats,
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


def stitch_overlapping_patches(
    patch_predictions: torch.Tensor,
    source_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    effective_size: tuple[int, int, int],
    weight_map: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch overlapping patches with Gaussian-weighted blending.

    这个函数用于直接拼接从模型输出的 overlapping patches。
    与 stitch_probability_patches 的区别是:
    - 原始 stitch_probability_patches 假设 patches 之间没有 overlap
    - 这个函数处理带有 overlap 的情况, 使用高斯权重融合

    Args:
        patch_predictions: Patch 预测结果, shape [N, 1, D, H, W] 或 [N, D, H, W]
        source_shape: 源体积形状 (D, H, W)
        patch_size: Patch 总尺寸
        effective_size: 有效预测区域尺寸
        weight_map: 可选的预计算权重图, shape 同 effective_size

    Returns:
        (fused_prediction, weight_sum): 融合后的预测和权重累加图
    """
    # 处理输入形状
    if patch_predictions.ndim == 4:
        patch_predictions = patch_predictions.unsqueeze(1)  # [N, 1, D, H, W]

    num_patches = patch_predictions.shape[0]

    # 计算偏移量
    offset_d = (patch_size[0] - effective_size[0]) // 2
    offset_h = (patch_size[1] - effective_size[1]) // 2
    offset_w = (patch_size[2] - effective_size[2]) // 2

    # 创建累加器
    fused = torch.zeros(source_shape, dtype=torch.float32)
    weight_sum = torch.zeros(source_shape, dtype=torch.float32)

    # 创建或使用权重图
    if weight_map is None:
        weight_map = torch.ones(effective_size, dtype=torch.float32)
    elif weight_map.shape != effective_size:
        raise ValueError(
            f"weight_map shape {weight_map.shape} must match effective_size {effective_size}"
        )

    # 生成 patch 位置
    from script.overlap_inference import generate_overlap_patch_positions
    patch_starts, _ = generate_overlap_patch_positions(
        source_shape,
        patch_size,
        effective_size,
        stride=effective_size,
        padding_mode="none",
    )

    # 累加
    for i, (z, y, x) in enumerate(patch_starts):
        if i >= num_patches:
            break

        pred = patch_predictions[i]
        while pred.ndim > 3:
            pred = pred[0]
        if pred.ndim == 3:
            pred = pred[0]  # Remove channel

        # 提取有效区域
        effective_pred = pred[
            offset_d : offset_d + effective_size[0],
            offset_h : offset_h + effective_size[1],
            offset_w : offset_w + effective_size[2],
        ]

        # 累加
        fused[
            z : z + effective_size[0],
            y : y + effective_size[1],
            x : x + effective_size[2],
        ] += effective_pred * weight_map

        weight_sum[
            z : z + effective_size[0],
            y : y + effective_size[1],
            x : x + effective_size[2],
        ] += weight_map

    # 归一化
    weight_sum = weight_sum.clamp(min=1e-8)
    fused = fused / weight_sum

    return fused, weight_sum
