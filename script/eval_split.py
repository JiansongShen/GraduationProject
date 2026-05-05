"""Evaluation utilities for segmentation models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import SimpleITK as sitk
import torch
from torch.utils.tensorboard import SummaryWriter

from core.config import TrainConfig
from data.MedicalPatchDataset import MedicalPatchDataset
from script.eval_utils import (
    align_target_shape,
    combine_to_nifti,
    combined_loss,
    dice_loss,
    sequential_patch_prediction,
)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    cfg=None,
    prediction_dir: Optional[Path] = None,
    store_single_res: bool = True,
) -> float:
    """Run evaluation on dataset and return average Dice score."""
    model.eval()
    total_loss, total_dice, num_batches = 0.0, 0.0, 0
    batch_size = cfg.train.batch_size if cfg else 1
    loss_kwargs = cfg.train.segmentation_loss_kwargs() if cfg else TrainConfig().segmentation_loss_kwargs()
    smooth = float(loss_kwargs.get("smooth", 1e-6))

    for sample_idx in range(len(dataset)):
        original_image = sitk.ReadImage(dataset.cases[sample_idx].image_path)
        image_tensor, label_src = dataset.get_src_item(sample_idx)
        images, labels = dataset.get_patches(sample_idx, sampling_mode="sequential")

        if labels is None:
            continue

        patch_list = []
        for patch_start in range(0, int(images.shape[0]), batch_size):
            patch_end = min(patch_start + batch_size, int(images.shape[0]))
            batch_images = images[patch_start:patch_end].to(device)
            batch_labels = labels[patch_start:patch_end].float().to(device)

            outputs = model(batch_images)
            batch_labels = align_target_shape(outputs, batch_labels)
            loss = combined_loss(outputs, batch_labels, **loss_kwargs)
            dice = 1.0 - dice_loss(outputs, batch_labels, smooth=smooth)

            total_loss += loss.item()
            total_dice += dice.item()
            num_batches += 1

            if store_single_res and sample_idx == 0:
                patch_list.extend([p.detach().cpu() for p in outputs[:, 0]])

        if store_single_res and sample_idx == 0 and prediction_dir:
            _save_predictions(
                prediction_dir, epoch, dataset, sample_idx, patch_list,
                image_tensor.shape, original_image
            )

    avg_dice = total_dice / max(num_batches, 1)
    avg_loss = total_loss / max(num_batches, 1)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Dice/val", avg_dice, epoch)
    logging.info("Validation - Loss: %.4f, Dice: %.4f", avg_loss, avg_dice)
    return avg_dice


def _save_predictions(
    prediction_dir: Path,
    epoch: int,
    dataset: MedicalPatchDataset,
    sample_idx: int,
    patch_list: list[torch.Tensor],
    src_shape: tuple,
    original_image: sitk.Image,
) -> None:
    """Save probability and binary predictions as NIfTI files."""
    prediction_dir.mkdir(parents=True, exist_ok=True)
    case_name = Path(dataset.cases[sample_idx].image_path).name

    for binarize, suffix in [(False, "probability"), (True, "binary")]:
        nifti = combine_to_nifti(
            patch_list, dataset.patch_size, src_shape,
            binarize=binarize, nifti_path=dataset.get_src_label_path(sample_idx)
        )
        nifti.CopyInformation(original_image)
        output_path = prediction_dir / f"epoch_{epoch + 1:04d}_case_{case_name}_eval_{suffix}.nii.gz"
        sitk.WriteImage(nifti, str(output_path))
        logging.info("Saved %s prediction to %s", suffix, output_path)


@torch.no_grad()
def sequential_predict(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    case_index: int,
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.Tensor], tuple, Path]:
    """Run deterministic patch inference for one case."""
    return sequential_patch_prediction(model, dataset, case_index=case_index, device=device, batch_size=batch_size)
