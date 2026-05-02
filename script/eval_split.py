from __future__ import annotations

import logging
from pathlib import Path

import SimpleITK as sitk
import torch
from torch.utils.tensorboard import SummaryWriter

from data.MedicalPatchDataset import MedicalPatchDataset
from model.aneurysm.model.AttentionUnet import AttentionUnet
from script.eval_utils import align_target_shape, combine_to_nifti, combined_loss, dice_loss


@torch.no_grad()
def evaluate(
        model: torch.nn.Module | AttentionUnet,
        dataset: MedicalPatchDataset,
        device: torch.device,
        epoch: int,
        writer: SummaryWriter,
        prediction_dir: Path | None = None,
        cfg=None,  # Add config parameter (optional to maintain compatibility)
        store_single_res: bool = True,
) -> float:
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    num_batches = 0

    total_volumes = len(dataset)
    logging.debug("Epoch %s validation started: total_volumes=%s", epoch + 1, total_volumes)

    for sample_idx in range(total_volumes):
        logging.debug("Epoch %s validation volume %s/%s loading", epoch + 1, sample_idx + 1, total_volumes)

        # Get original image for spacing restoration
        original_image_itk = sitk.ReadImage(dataset.cases[sample_idx].image_path)
        original_spacing = original_image_itk.GetSpacing()
        original_direction = original_image_itk.GetDirection()
        original_origin = original_image_itk.GetOrigin()

        # Load resampled image and label for processing
        image_tensor, label_src = dataset.get_src_item(sample_idx)
        images, labels = dataset.get_patches(sample_idx, sampling_mode="sequential")
        patch_list: list[torch.Tensor] = []
        if labels is None:
            logging.warning("Epoch %s validation volume %s/%s has no labels. Skipping.", epoch + 1, sample_idx + 1,
                            total_volumes)
            continue

        patch_count = int(images.shape[0])
        logging.debug(
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

            output_prob = torch.sigmoid(outputs)
            pred_binary = (output_prob > 0.5).float()
            output_min = float(output_prob.min().item())
            output_max = float(output_prob.max().item())
            output_mean = float(output_prob.mean().item())
            output_positive_ratio = float(pred_binary.mean().item())
            target_positive_ratio = float((batch_labels > 0.5).float().mean().item())
            if store_single_res and sample_idx == 0:
                patch_list.extend([patch.detach().cpu() for patch in output_prob[:, 0]])

            dice = 1.0 - dice_loss(output_prob, batch_labels)

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            num_batches += 1

            avg_loss = epoch_loss / num_batches
            avg_dice = epoch_dice / num_batches
            logging.info(
                "Epoch %s validation volume %s/%s sequential patch batch %s-%s done: loss=%.6f dice=%.6f avg_loss=%.6f avg_dice=%.6f prob_min=%.6f prob_max=%.6f prob_mean=%.6f pred_positive_ratio=%.6f target_positive_ratio=%.6f global_patch_step=%s",
                epoch + 1,
                sample_idx + 1,
                total_volumes,
                patch_start_idx + 1,
                patch_end_idx,
                loss.item(),
                dice.item(),
                avg_loss,
                avg_dice,
                output_min,
                output_max,
                output_mean,
                output_positive_ratio,
                target_positive_ratio,
                num_batches,
            )

        if store_single_res and sample_idx == 0 and prediction_dir is not None:
            src_shape_tuple: tuple[int, int, int] = image_tensor.shape[0], image_tensor.shape[1], image_tensor.shape[2]
            if len(src_shape_tuple) != 3:
                raise ValueError(f"Invalid eval source shape: {src_shape_tuple}")
            prediction_dir.mkdir(parents=True, exist_ok=True)
            if patch_list:
                probability_nifti = combine_to_nifti(patch_list, dataset.patch_size, src_shape_tuple, binarize=False,
                                                     nifti_path=dataset.get_src_label_path(sample_idx))

                # Restore original spacing to the reconstructed image
                probability_nifti.SetSpacing(original_spacing)
                probability_nifti.SetDirection(original_direction)
                probability_nifti.SetOrigin(original_origin)

                probability_path = prediction_dir / f"epoch_{epoch + 1:04d}_case_{Path(dataset.cases[sample_idx].image_path).name}_eval_probability.nii.gz"
                sitk.WriteImage(probability_nifti, str(probability_path))

                binary_nifti = combine_to_nifti(patch_list, dataset.patch_size, src_shape_tuple, binarize=True,
                                                nifti_path=dataset.get_src_label_path(sample_idx))

                # Restore original spacing to the reconstructed binary image
                binary_nifti.SetSpacing(original_spacing)
                binary_nifti.SetDirection(original_direction)
                binary_nifti.SetOrigin(original_origin)

                binary_path = prediction_dir / f"epoch_{epoch + 1:04d}_case_{Path(dataset.cases[sample_idx].image_path).name}_eval_binary.nii.gz"
                sitk.WriteImage(binary_nifti, str(binary_path))
                logging.info("Saved eval probability prediction to %s", probability_path)
                logging.info("Saved eval binary prediction to %s", binary_path)

    avg_loss = epoch_loss / max(num_batches, 1)
    avg_dice = epoch_dice / max(num_batches, 1)

    writer.add_scalar("Loss/val", avg_loss, epoch)
    writer.add_scalar("Dice/val", avg_dice, epoch)

    logging.info(f"Validation - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    return avg_dice


def evaluate_full_pipeline(
        model: torch.nn.Module,
        dataset: MedicalPatchDataset,
        device: torch.device,
        epoch: int,
        writer: SummaryWriter,
        prediction_dir: Path | None = None,
        cfg=None,
        binarize_result: bool = False,
) -> float:
    """
    Full evaluation pipeline with proper spacing handling:
    1. Spacing normalization
    2. Sequential patching
    3. Individual patch prediction
    4. Combining patches back to nifti
    5. Spacing restoration
    6. Save results
    """
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    num_batches = 0

    total_volumes = len(dataset)
    logging.info("Epoch %s validation started (full pipeline): total_volumes=%s", epoch + 1, total_volumes)

    for sample_idx in range(total_volumes):
        logging.info("Epoch %s validation volume %s/%s loading (full pipeline)", epoch + 1, sample_idx + 1,
                     total_volumes)

        # Step 1: Get original image metadata for spacing restoration later
        original_image_itk = sitk.ReadImage(dataset.cases[sample_idx].image_path)
        original_spacing = original_image_itk.GetSpacing()
        original_direction = original_image_itk.GetDirection()
        original_origin = original_image_itk.GetOrigin()

        # Step 2: Load and resample the case data (spacing normalization happens in dataset)
        image_tensor, label_tensor = dataset.get_src_item(sample_idx)
        images, labels = dataset.get_patches(sample_idx, sampling_mode="sequential")

        if labels is None:
            logging.warning("Epoch %s validation volume %s/%s has no labels. Skipping.", epoch + 1, sample_idx + 1,
                            total_volumes)
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

        # Step 3: Process patches individually
        all_outputs = []
        batch_size = cfg.train.batch_size if cfg is not None else 1
        for patch_start_idx in range(0, patch_count, batch_size):
            patch_end_idx = min(patch_start_idx + batch_size, patch_count)

            batch_images = images[patch_start_idx:patch_end_idx].to(device)
            batch_labels = labels[patch_start_idx:patch_end_idx].float().to(device)

            outputs = model(batch_images)
            batch_labels = align_target_shape(outputs, batch_labels)
            loss = combined_loss(outputs, batch_labels)

            output_prob = torch.sigmoid(outputs)
            pred_binary = (output_prob > 0.5).float()
            dice = 1.0 - dice_loss(output_prob, batch_labels)

            output_min = float(output_prob.min().item())
            output_max = float(output_prob.max().item())
            output_mean = float(output_prob.mean().item())
            output_positive_ratio = float(pred_binary.mean().item())
            target_positive_ratio = float((batch_labels > 0.5).float().mean().item())

            epoch_loss += loss.item()
            epoch_dice += dice.item()
            num_batches += 1

            logging.info(
                "Full pipeline epoch %s volume %s/%s patch batch %s-%s done: loss=%.6f dice=%.6f prob_min=%.6f prob_max=%.6f prob_mean=%.6f pred_positive_ratio=%.6f target_positive_ratio=%.6f",
                epoch + 1,
                sample_idx + 1,
                total_volumes,
                patch_start_idx + 1,
                patch_end_idx,
                loss.item(),
                dice.item(),
                output_min,
                output_max,
                output_mean,
                output_positive_ratio,
                target_positive_ratio,
            )

            # Store predictions for later combination
            for output_tensor in output_prob:
                all_outputs.append(output_tensor.cpu())

        # Step 4: Combine patches back to full nifti volume
        src_shape_tuple = tuple(int(dim) for dim in image_tensor.shape)
        if len(src_shape_tuple) != 3:
            raise ValueError(f"Invalid eval source shape: {src_shape_tuple}")

        combined_prediction = combine_to_nifti(
            all_outputs,
            dataset.patch_size,
            src_shape_tuple,
            binarize=binarize_result,
            nifti_path=Path(dataset.cases[sample_idx].image_path),
        )
        logging.info(
            "Combined full pipeline prediction: case=%s patch_count=%s binarize=%s src_shape=%s",
            Path(dataset.cases[sample_idx].image_path).name,
            len(all_outputs),
            binarize_result,
            src_shape_tuple,
        )

        # Step 5: Restore original spacing to the combined prediction
        combined_prediction.SetSpacing(original_spacing)
        combined_prediction.SetDirection(original_direction)
        combined_prediction.SetOrigin(original_origin)

        # Step 6: Save the final result with original spacing
        if prediction_dir:
            prediction_dir.mkdir(parents=True, exist_ok=True)
            result_filename = f"epoch_{epoch + 1:04d}_case_{sample_idx:04d}_prediction.nii.gz"
            result_path = prediction_dir / result_filename
            sitk.WriteImage(combined_prediction, str(result_path))
            logging.info("Saved full pipeline prediction to %s", result_path)

    avg_loss = epoch_loss / max(num_batches, 1)
    avg_dice = epoch_dice / max(num_batches, 1)

    writer.add_scalar("Loss/val_full", avg_loss, epoch)
    writer.add_scalar("Dice/val_full", avg_dice, epoch)

    logging.info(f"Full Pipeline Validation - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    return avg_dice
