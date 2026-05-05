from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import SimpleITK as sitk
import torch
import torch.nn.functional as F

from data.MedicalPatchDataset import MedicalPatchDataset

if TYPE_CHECKING:
    from script.overlap_inference import OverlapInferenceConfig


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.float().reshape(-1)
    target_flat = target.float().reshape(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1.0 - dice_coeff


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1e-6,
) -> torch.Tensor:
    pred_flat = pred.float().reshape(-1)
    target_flat = target.float().reshape(-1)
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1.0 - target_flat)).sum()
    fn = ((1.0 - pred_flat) * target_flat).sum()
    return 1.0 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    smooth: float = 1e-6,
) -> torch.Tensor:
    pred = pred.float().clamp(min=smooth, max=1.0 - smooth)
    target = target.float()
    pt = torch.where(target > 0.5, pred, 1.0 - pred)
    return (-(1.0 - pt).pow(gamma) * torch.log(pt)).mean()


def combined_loss_with_parts(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str = "tversky_bce",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_gamma: float = 2.0,
    smooth: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bce = F.binary_cross_entropy(pred_prob.float().clamp(0.0, 1.0), target.float())
    loss_type = loss_type.lower()
    if loss_type == "tversky_bce":
        shape_term = tversky_loss(pred_prob, target, alpha=tversky_alpha, beta=tversky_beta, smooth=smooth)
    elif loss_type == "focal_bce":
        shape_term = focal_loss(pred_prob, target, gamma=focal_gamma, smooth=smooth)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return bce + shape_term, bce, shape_term


def combined_loss(pred_prob: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
    total, _, _ = combined_loss_with_parts(pred_prob, target, **kwargs)
    return total


def align_target_shape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == pred.ndim - 1:
        target = target.unsqueeze(1)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
    return target.float().clamp(0.0, 1.0)


def can_stitch_exactly(src_shape: tuple[int, int, int], patch_size: tuple[int, int, int]) -> bool:
    return all(dim % patch == 0 for dim, patch in zip(src_shape, patch_size))


def stitch_probability_patches(
    patch_list: list[torch.Tensor],
    patch_shape: tuple[int, int, int],
    src_shape: tuple[int, int, int],
    *,
    binarize: bool = False,
) -> tuple[sitk.Image, dict[str, float]]:
    if not patch_list:
        raise ValueError("Cannot stitch empty patch list")

    combined = torch.zeros(src_shape, dtype=torch.float32)
    idx = 0
    for z in range(0, src_shape[0], patch_shape[0]):
        for y in range(0, src_shape[1], patch_shape[1]):
            for x in range(0, src_shape[2], patch_shape[2]):
                if idx >= len(patch_list):
                    raise ValueError(f"Patch count mismatch: expected more than {idx}, got {len(patch_list)}")
                patch = patch_list[idx].detach().cpu().float()
                while patch.ndim > 3:
                    patch = patch[0]
                z_end = min(z + patch_shape[0], src_shape[0])
                y_end = min(y + patch_shape[1], src_shape[1])
                x_end = min(x + patch_shape[2], src_shape[2])
                combined[z:z_end, y:y_end, x:x_end] = patch[: z_end - z, : y_end - y, : x_end - x].clamp(0.0, 1.0)
                idx += 1

    if idx != len(patch_list):
        raise ValueError(f"Patch count mismatch: expected {idx}, got {len(patch_list)}")

    if binarize:
        combined = (combined > 0.5).float()
    image = sitk.GetImageFromArray(combined.numpy().astype("uint8" if binarize else "float32"))
    return image, {"patches_used": float(len(patch_list))}


def combine_to_nifti(
    unified_spacing_patch_list: list[torch.Tensor],
    patch_shape: tuple[int, int, int],
    src_shape: tuple[int, int, int],
    *,
    binarize: bool = False,
    nifti_path: Path,
) -> sitk.Image:
    dataset_like_image = sitk.ReadImage(str(nifti_path))
    resampled_prediction, _ = stitch_probability_patches(
        unified_spacing_patch_list,
        patch_shape,
        src_shape,
        binarize=binarize,
    )
    resampled_prediction.CopyInformation(dataset_like_image)
    return resampled_prediction


def restore_prediction_to_source_grid(
    prediction_train_space: sitk.Image,
    original_image: sitk.Image,
    *,
    binarize: bool,
) -> sitk.Image:
    interpolator = sitk.sitkNearestNeighbor if binarize else sitk.sitkLinear
    pixel_type = sitk.sitkUInt8 if binarize else sitk.sitkFloat32
    restored = sitk.Resample(
        prediction_train_space,
        original_image,
        sitk.Transform(),
        interpolator,
        0.0,
        pixel_type,
    )
    restored.CopyInformation(original_image)
    return restored


def sequential_patch_prediction(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    *,
    case_index: int,
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.Tensor], tuple[int, int, int], Path, tuple[float, float, float], tuple[float, ...], tuple[float, float, float]]:
    case = dataset.cases[case_index]
    original_image = sitk.ReadImage(case.image_path)
    image_tensor, _ = dataset.get_src_item(case_index)
    images, _ = dataset.get_patches(case_index, sampling_mode="sequential")
    if int(images.shape[0]) == 0:
        raise ValueError(f"No sequential patches available for case: {case.image_path}")

    probability_patches: list[torch.Tensor] = []
    with torch.no_grad():
        for patch_start in range(0, int(images.shape[0]), batch_size):
            batch = images[patch_start : patch_start + batch_size].to(device)
            outputs = model(batch).detach().cpu()
            for output in outputs:
                while output.ndim > 3:
                    output = output[0]
                probability_patches.append(output)

    return (
        probability_patches,
        tuple(int(d) for d in image_tensor.shape),
        Path(case.image_path),
        original_image.GetSpacing(),
        original_image.GetDirection(),
        original_image.GetOrigin(),
    )


def overlap_patch_prediction(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    *,
    case_index: int,
    device: torch.device,
    overlap_config: "OverlapInferenceConfig",
) -> tuple[torch.Tensor, tuple[int, int, int], Path, tuple[float, float, float], tuple[float, ...], tuple[float, float, float], dict]:
    from script.overlap_inference import predict_with_overlap

    case = dataset.cases[case_index]
    original_image = sitk.ReadImage(case.image_path)
    image_tensor, _ = dataset.get_src_item(case_index)
    prediction = predict_with_overlap(
        model=model,
        image=image_tensor.unsqueeze(0).unsqueeze(0).float(),
        config=overlap_config,
        device=device,
    )
    pred_np = prediction.squeeze().cpu().numpy()
    return (
        prediction.squeeze(0),
        tuple(int(d) for d in image_tensor.shape),
        Path(case.image_path),
        original_image.GetSpacing(),
        original_image.GetDirection(),
        original_image.GetOrigin(),
        {
            "prob_min": float(pred_np.min()),
            "prob_max": float(pred_np.max()),
            "prob_mean": float(pred_np.mean()),
            "prob_std": float(pred_np.std()),
            "inference_mode": "effective_center",
        },
    )
