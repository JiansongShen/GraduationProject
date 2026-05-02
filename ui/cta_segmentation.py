"""Server-side CTA volume segmentation using the same spacing and patching rules as training.

Pipeline (matches validation in ``script/eval_split.py``):

1. Record the uploaded volume's original geometry (spacing, direction, origin).
2. Resample internally to the model's target spacing (via ``MedicalPatchDataset`` / ``resample_in_memory``).
3. Tile the resampled volume into fixed-size patches in deterministic sequential order.
4. Run the Attention U-Net on each patch batch.
5. Stitch logits into one volume on the resampled grid.
6. Copy the original geometry onto the stitched result so the mask aligns with the uploaded scan in physical space.
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
from pathlib import Path

import SimpleITK as sitk
import torch

from core.config import Config, DataConfig
from data.MedicalPatchDataset import MedicalPatchDataset
from script.eval_split import combine_to_nifti
from script.train_split import build_model

logger = logging.getLogger("gradulate.segmentation")


def inference_case_filename(file_patterns: list[str], upload_identifier: str) -> str:
    """Build a unique filename that still matches ``DataConfig.file_patterns`` (e.g. ``*_origin.nii.gz``)."""
    if not file_patterns:
        return f"{upload_identifier}_model_input_origin.nii.gz"
    pattern = file_patterns[0]
    if "*" in pattern:
        literal_suffix = pattern[pattern.index("*") + 1 :]
        return f"{upload_identifier}_model_input{literal_suffix}"
    return f"{upload_identifier}_model_input_{pattern}"


def _resolve_path_under_application_root(application_root: Path, user_path: Path) -> Path:
    """Turn user-supplied paths into absolute paths, defaulting relative paths to the project root."""
    expanded = user_path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (application_root / expanded).resolve()


def build_dataset_for_single_uploaded_volume(
    *,
    volume_path_on_disk: Path,
    data_settings: DataConfig,
    patch_size: tuple[int, int, int],
    random_seed: int,
) -> MedicalPatchDataset:
    """Point the patch dataset at one NIfTI file in the same folder (no label required on disk)."""
    inference_directory = str(volume_path_on_disk.parent)
    inference_data_settings = dataclasses.replace(
        data_settings,
        train_dirs=[inference_directory],
        eval_dirs=[],
        patch_sampling_mode="sequential",
        patches_per_volume=1_000_000,
        max_load=1,
    )
    return MedicalPatchDataset(
        cfg=inference_data_settings,
        patch_size=patch_size,
        seed=random_seed,
    )


def load_trained_segmentation_weights(
    *,
    model: torch.nn.Module,
    checkpoint_file: Path,
    device: torch.device,
) -> None:
    """Load ``model_state_dict`` from a training checkpoint saved by ``script.train_split``."""
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")
    try:
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location=device)
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    model.load_state_dict(state)
    model.eval()


def run_patch_based_segmentation(
    *,
    configuration: Config,
    checkpoint_file: Path,
    input_volume_path: Path,
    probability_output_path: Path,
    binary_mask_output_path: Path,
    device: torch.device,
) -> None:
    """Full spacing-aware segmentation: resample → patch → predict → stitch → restore geometry → write NIfTI."""
    patch_size = tuple(int(dim) for dim in configuration.train.patch_size)

    # 1) Original geometry from the file the user uploaded (before any resampling).
    original_volume = sitk.ReadImage(str(input_volume_path))
    original_spacing = original_volume.GetSpacing()
    original_direction = original_volume.GetDirection()
    original_origin = original_volume.GetOrigin()

    # 2–3) Dataset applies target spacing and exposes sequential patches (same as validation).
    dataset = build_dataset_for_single_uploaded_volume(
        volume_path_on_disk=input_volume_path,
        data_settings=configuration.data,
        patch_size=patch_size,
        random_seed=configuration.seed,
    )
    if len(dataset.cases) == 0:
        raise RuntimeError("No volume matched file_patterns; check data.file_patterns and the inference filename.")

    input_resolved = input_volume_path.resolve()
    matching_case_indices = [
        index
        for index, case in enumerate(dataset.cases)
        if Path(case.image_path).resolve() == input_resolved
    ]
    if not matching_case_indices:
        raise RuntimeError(
            f"No dataset case matches the inference file {input_volume_path}. "
            f"Found {len(dataset.cases)} other case(s) under the same directory; use a per-request folder with only this volume."
        )
    case_index = matching_case_indices[0]
    case_image_path = Path(dataset.cases[case_index].image_path)

    resampled_shape_tensor, _optional_label = dataset.get_src_item(case_index)
    patch_batch_images, label_batch = dataset.get_patches(case_index, sampling_mode="sequential")
    if label_batch is not None:
        logger.info("A label file was found next to the inference volume; it is ignored for pure inference.")

    patch_count = int(patch_batch_images.shape[0])
    if patch_count == 0:
        raise RuntimeError("No patches were produced for this volume.")

    model = build_model(configuration, device)
    load_trained_segmentation_weights(model=model, checkpoint_file=checkpoint_file, device=device)

    # 4) Per-patch predictions.
    patch_probability_tensors: list[torch.Tensor] = []
    batch_limit = max(1, configuration.train.batch_size)

    with torch.no_grad():
        for batch_start in range(0, patch_count, batch_limit):
            batch_end = min(batch_start + batch_limit, patch_count)
            batch_images = patch_batch_images[batch_start:batch_end].to(device)
            network_outputs = model(batch_images)
            for single_output in network_outputs:
                patch_probability_tensors.append(single_output.detach().cpu())

    # 5) Stitch on the resampled voxel grid (numpy indexing matches the dataset).
    source_shape = tuple(int(dim) for dim in resampled_shape_tensor.shape)
    if len(source_shape) != 3:
        raise ValueError(f"Expected a rank-3 resampled volume, got shape {source_shape}.")

    probability_volume = combine_to_nifti(
        patch_probability_tensors,
        patch_size,
        source_shape,
        binarize=False,
        nifti_path=case_image_path,
    )
    binary_mask_volume = combine_to_nifti(
        patch_probability_tensors,
        patch_size,
        source_shape,
        binarize=True,
        nifti_path=case_image_path,
    )

    # 6) Restore physical space so SimpleITK viewers overlay mask on the original CTA.
    for volume in (probability_volume, binary_mask_volume):
        volume.SetSpacing(original_spacing)
        volume.SetDirection(original_direction)
        volume.SetOrigin(original_origin)

    probability_output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(probability_volume, str(probability_output_path))
    sitk.WriteImage(binary_mask_volume, str(binary_mask_output_path))
    logger.info("Wrote probability map to %s", probability_output_path)
    logger.info("Wrote binary mask to %s", binary_mask_output_path)


def prepare_uploaded_volume_for_dataset(
    *,
    uploaded_file: Path,
    destination_directory: Path,
    file_patterns: list[str],
    upload_identifier: str,
) -> Path:
    """Copy the upload to a name that matches ``file_patterns`` inside ``destination_directory``."""
    destination_directory.mkdir(parents=True, exist_ok=True)
    target_name = inference_case_filename(file_patterns, upload_identifier)
    target_path = destination_directory / target_name
    shutil.copyfile(uploaded_file, target_path)
    return target_path
