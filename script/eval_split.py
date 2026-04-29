from __future__ import annotations

import logging
from pathlib import Path

import SimpleITK as sitk
import torch
from torch.utils.tensorboard import SummaryWriter

from data.MedicalPatchDataset import MedicalPatchDataset


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    dice_coeff = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1.0 - dice_coeff


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def align_target_shape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.ndim == pred.ndim - 1:
        target = target.unsqueeze(1)
    if target.shape != pred.shape:
        raise ValueError(
            f"Prediction/target shape mismatch after alignment: pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    return target


def combine_to_nifti(
    patch_list: list[torch.Tensor],
    patch_shape: tuple[int, int, int],
    src_shape: tuple[int, int, int],
) -> sitk.Image:
    """Stitch sequential prediction patches back to one volume."""
    if not patch_list:
        raise ValueError("Cannot stitch eval prediction: patch_list is empty.")

    patch_d, patch_h, patch_w = patch_shape
    src_d, src_h, src_w = src_shape
    combined = torch.zeros(src_shape, dtype=torch.float32)
    patch_idx = 0

    for z in range(0, src_d - patch_d + 1, patch_d):
        for y in range(0, src_h - patch_h + 1, patch_h):
            for x in range(0, src_w - patch_w + 1, patch_w):
                if patch_idx >= len(patch_list):
                    return sitk.GetImageFromArray(combined.numpy())

                patch = patch_list[patch_idx].detach().cpu()
                if patch.ndim == 5:
                    patch = patch[0, 0]
                elif patch.ndim == 4:
                    patch = patch[0]
                if tuple(patch.shape) != patch_shape:
                    raise ValueError(f"Invalid eval patch shape: {tuple(patch.shape)}, expected {patch_shape}")

                combined[z:z + patch_d, y:y + patch_h, x:x + patch_w] = patch.float()
                patch_idx += 1

    return sitk.GetImageFromArray(combined.numpy())


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    prediction_dir: Path | None = None,
    cfg = None,  # Add config parameter (optional to maintain compatibility)
    store_single_res: bool = True,
) -> float:
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    num_batches = 0

    total_volumes = len(dataset)
    logging.info("Epoch %s validation started: total_volumes=%s", epoch + 1, total_volumes)

    for sample_idx in range(total_volumes):
        logging.info("Epoch %s validation volume %s/%s loading", epoch + 1, sample_idx + 1, total_volumes)
        _, label_src = dataset.get_src_item(sample_idx)
        images, labels = dataset[sample_idx]
        patch_list: list[torch.Tensor] = []
        if labels is None:
            logging.warning("Epoch %s validation volume %s/%s has no labels. Skipping.", epoch + 1, sample_idx + 1, total_volumes)
            continue

        patch_count = int(images.shape[0])
        logging.info(
            "Epoch %s validation volume %s/%s loaded: patches=%s image_shape=%s label_shape=%s",
            epoch + 1,
            sample_idx + 1,
            total_volumes,
            patch_count,
            tuple(images.shape),
            tuple(labels.shape),
        )

        # Process patches in batches during evaluation
        batch_size = cfg.train.batch_size if cfg is not None else 1  # Use config batch size if available
        for patch_start_idx in range(0, patch_count, batch_size):
            patch_end_idx = min(patch_start_idx + batch_size, patch_count)
            
            batch_images = images[patch_start_idx:patch_end_idx].to(device)
            batch_labels = labels[patch_start_idx:patch_end_idx].float().to(device)

            outputs = model(batch_images)
            batch_labels = align_target_shape(outputs, batch_labels)
            loss = combined_loss(outputs, batch_labels)

            # If storing single results and this is the first volume, store outputs
            if store_single_res and sample_idx == 0:
                patch_list.extend([output_tensor.unsqueeze(0) for output_tensor in outputs])

            pred_binary = (outputs > 0.5).float()
            dice = 1.0 - dice_loss(pred_binary, batch_labels)

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            num_batches += 1

            avg_loss = epoch_loss / num_batches
            avg_dice = epoch_dice / num_batches
            logging.info(
                "Epoch %s validation volume %s/%s patch batch %s-%s done: loss=%.6f dice=%.6f avg_loss=%.6f avg_dice=%.6f global_patch_step=%s",
                epoch + 1,
                sample_idx + 1,
                total_volumes,
                patch_start_idx + 1,
                patch_end_idx,
                loss.item(),
                dice.item(),
                avg_loss,
                avg_dice,
                num_batches,
            )

        if store_single_res and sample_idx == 0 and prediction_dir is not None:
            src_shape_tuple = tuple(int(dim) for dim in label_src.shape)
            if len(src_shape_tuple) != 3:
                raise ValueError(f"Invalid eval source shape: {src_shape_tuple}")
            prediction_dir.mkdir(parents=True, exist_ok=True)
            # Combine only the first batch of patches for visualization
            if patch_list:
                # Take the first patch from the batch for visualization
                first_patch = patch_list[0].squeeze(0).unsqueeze(0)  # Shape: [1, D, H, W]
                nifti = combine_to_nifti([first_patch], dataset.patch_size, src_shape_tuple)
                prediction_path = prediction_dir / f"epoch_{epoch + 1:04d}_case_{sample_idx:04d}_eval_prediction.nii.gz"
                sitk.WriteImage(nifti, str(prediction_path))
                logging.info("Saved eval sample prediction to %s", prediction_path)

    avg_loss = epoch_loss / max(num_batches, 1)
    avg_dice = epoch_dice / max(num_batches, 1)

    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Dice/val", avg_dice, epoch)

    logging.info(f"Validation - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    return avg_dice
