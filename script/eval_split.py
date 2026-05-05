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
    can_stitch_exactly,
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
    model.eval()
    total_loss, total_dice, num_batches = 0.0, 0.0, 0
    batch_size = cfg.train.batch_size if cfg else 1
    loss_kwargs = cfg.train.segmentation_loss_kwargs() if cfg else TrainConfig().segmentation_loss_kwargs()
    smooth = float(loss_kwargs.get("smooth", 1e-6))

    for sample_idx in range(len(dataset)):
        original_image = sitk.ReadImage(dataset.cases[sample_idx].image_path)
        image_tensor, _ = dataset.get_src_item(sample_idx)
        images, labels = dataset.get_patches(sample_idx, sampling_mode="sequential")

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
                prediction_dir,
                epoch,
                dataset,
                sample_idx,
                patch_list,
                tuple(int(v) for v in image_tensor.shape),
                original_image,
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
    prediction_dir.mkdir(parents=True, exist_ok=True)
    case_name = Path(dataset.cases[sample_idx].image_path).name
    patch_shape = tuple(int(v) for v in dataset.patch_size)
    src_shape_3d = tuple(int(v) for v in src_shape)

    if not patch_list:
        raise ValueError("Cannot save prediction snapshot: no patches produced")

    exact = can_stitch_exactly(src_shape_3d, patch_shape)
    expected_count = 1
    for dim, patch in zip(src_shape_3d, patch_shape):
        expected_count *= dim // patch if exact else (dim + patch - 1) // patch
    if len(patch_list) != expected_count:
        raise ValueError(
            f"Cannot reconstruct snapshot: patch_count={len(patch_list)}, expected={expected_count}, "
            f"src_shape={src_shape_3d}, patch_shape={patch_shape}"
        )
    if not exact:
        logging.warning(
            "Snapshot patches cannot tile exactly; using zero-padded edge crop. src_shape=%s patch_shape=%s",
            src_shape_3d,
            patch_shape,
        )

    canvas = torch.zeros(src_shape_3d, dtype=torch.float32)
    idx = 0
    for z in range(0, src_shape_3d[0], patch_shape[0]):
        for y in range(0, src_shape_3d[1], patch_shape[1]):
            for x in range(0, src_shape_3d[2], patch_shape[2]):
                patch = patch_list[idx].detach().cpu().float()
                while patch.ndim > 3:
                    patch = patch[0]
                z_end = min(z + patch_shape[0], src_shape_3d[0])
                y_end = min(y + patch_shape[1], src_shape_3d[1])
                x_end = min(x + patch_shape[2], src_shape_3d[2])
                canvas[z:z_end, y:y_end, x:x_end] = patch[: z_end - z, : y_end - y, : x_end - x].clamp(0.0, 1.0)
                idx += 1

    for binarize, suffix in [(False, "probability"), (True, "binary")]:
        arr = (canvas > 0.5).numpy().astype("uint8") if binarize else canvas.numpy().astype("float32")
        train_space = sitk.GetImageFromArray(arr)
        train_space.SetSpacing(tuple(float(v) for v in dataset.target_spacing))
        train_space.SetOrigin(original_image.GetOrigin())
        train_space.SetDirection(original_image.GetDirection())
        restored = dataset.restore_to_original_space(train_space, sample_idx, is_label=binarize)
        restored.CopyInformation(original_image)
        output_path = prediction_dir / f"epoch_{epoch + 1:04d}_case_{case_name}_{suffix}.nii.gz"
        sitk.WriteImage(restored, str(output_path))
        logging.info("Saved %s prediction to %s", suffix, output_path)


@torch.no_grad()
def sequential_predict(
    model: torch.nn.Module,
    dataset: MedicalPatchDataset,
    case_index: int,
    device: torch.device,
    batch_size: int,
) -> tuple[list[torch.Tensor], tuple, Path]:
    return sequential_patch_prediction(model, dataset, case_index=case_index, device=device, batch_size=batch_size)
