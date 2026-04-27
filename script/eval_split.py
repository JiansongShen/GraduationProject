from __future__ import annotations

import logging
from pathlib import Path

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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> float:
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    num_batches = 0

    total_volumes = len(dataset)
    logging.info("Epoch %s validation started: total_volumes=%s", epoch + 1, total_volumes)

    for sample_idx in range(total_volumes):
        logging.info("Epoch %s validation volume %s/%s loading", epoch + 1, sample_idx + 1, total_volumes)
        images, labels = dataset[sample_idx]
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

        for patch_idx in range(patch_count):
            logging.info(
                "Epoch %s validation volume %s/%s patch %s/%s started",
                epoch + 1,
                sample_idx + 1,
                total_volumes,
                patch_idx + 1,
                patch_count,
            )
            patch_images = images[patch_idx : patch_idx + 1].to(device)
            patch_labels = labels[patch_idx : patch_idx + 1].float().to(device)

            outputs = model(patch_images)
            patch_labels = align_target_shape(outputs, patch_labels)
            loss = combined_loss(outputs, patch_labels)

            pred_binary = (outputs > 0.5).float()
            dice = 1.0 - dice_loss(pred_binary, patch_labels)

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            num_batches += 1

            avg_loss = epoch_loss / num_batches
            avg_dice = epoch_dice / num_batches
            logging.info(
                "Epoch %s validation volume %s/%s patch %s/%s done: loss=%.6f dice=%.6f avg_loss=%.6f avg_dice=%.6f global_patch_step=%s",
                epoch + 1,
                sample_idx + 1,
                total_volumes,
                patch_idx + 1,
                patch_count,
                loss.item(),
                dice.item(),
                avg_loss,
                avg_dice,
                num_batches,
            )

    avg_loss = epoch_loss / max(num_batches, 1)
    avg_dice = epoch_dice / max(num_batches, 1)

    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Dice/val", avg_dice, epoch)

    logging.info(f"Validation - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    return avg_dice
