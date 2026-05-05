from __future__ import annotations

import dataclasses
import logging
import shutil
from pathlib import Path

import SimpleITK as sitk
import torch

from core.config import Config, DataConfig
from data.MedicalPatchDataset import MedicalPatchDataset
from script.eval_utils import restore_prediction_to_source_grid
from script.overlap_inference import OverlapInferenceConfig, predict_with_overlap
from script.train_split import build_model

logger = logging.getLogger("gradulate.segmentation")


def inference_case_filename(_file_patterns: list[str], upload_identifier: str) -> str:
    return f"{upload_identifier}_origin.nii.gz"


def build_dataset_for_single_uploaded_volume(
    *,
    volume_path_on_disk: Path,
    data_settings: DataConfig,
    patch_size: tuple[int, int, int],
    random_seed: int,
) -> MedicalPatchDataset:
    inference_directory = str(volume_path_on_disk.parent)
    inference_data_settings = dataclasses.replace(
        data_settings,
        train_dirs=[inference_directory],
        eval_dirs=[],
        patch_sampling_mode="sequential",
        patches_per_volume=1_000_000,
        max_load=1,
    )
    label_path = volume_path_on_disk.with_name(volume_path_on_disk.name.replace(data_settings.origin_suffix, data_settings.label_suffix))
    if not label_path.exists():
        sitk.WriteImage(sitk.Image(sitk.ReadImage(str(volume_path_on_disk)).GetSize(), sitk.sitkUInt8), str(label_path))
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
) -> dict:
    patch_size = tuple(int(dim) for dim in configuration.inference.patch_size)
    effective_size = tuple(int(dim) for dim in configuration.inference.effective_size)

    original_volume = sitk.ReadImage(str(input_volume_path))
    dataset = build_dataset_for_single_uploaded_volume(
        volume_path_on_disk=input_volume_path,
        data_settings=configuration.data,
        patch_size=patch_size,
        random_seed=configuration.seed,
    )
    if len(dataset.cases) == 0:
        raise RuntimeError("No valid inference volume found")

    case_index = 0
    image_tensor, _ = dataset.get_src_item(case_index)

    model = build_model(configuration, device)
    load_trained_segmentation_weights(model=model, checkpoint_file=checkpoint_file, device=device)

    inference_cfg = OverlapInferenceConfig(
        enabled=True,
        patch_size=patch_size,
        effective_size=effective_size,
        batch_size=configuration.inference.batch_size,
        use_amp=configuration.inference.use_amp,
    )

    with torch.no_grad():
        prediction = predict_with_overlap(
            model=model,
            image=image_tensor.unsqueeze(0).unsqueeze(0).float(),
            config=inference_cfg,
            device=device,
        )

    pred_np = prediction.squeeze().cpu().numpy().astype("float32")
    train_space_prob = sitk.GetImageFromArray(pred_np)
    label_like = sitk.ReadImage(str(dataset.get_src_label_path(case_index)))
    train_space_prob.CopyInformation(label_like)

    restored_prob = dataset.restore_to_original_space(train_space_prob, case_index, is_label=False)
    restored_prob = restore_prediction_to_source_grid(restored_prob, original_volume, binarize=False)

    train_space_bin = sitk.GetImageFromArray((pred_np > 0.5).astype("uint8"))
    train_space_bin.CopyInformation(label_like)
    restored_bin = dataset.restore_to_original_space(train_space_bin, case_index, is_label=True)
    restored_bin = restore_prediction_to_source_grid(restored_bin, original_volume, binarize=True)

    probability_output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(restored_prob, str(probability_output_path))
    sitk.WriteImage(restored_bin, str(binary_mask_output_path))

    binary_arr = sitk.GetArrayFromImage(restored_bin)
    return {
        "inference_mode": "effective_center",
        "patch_size": patch_size,
        "effective_size": effective_size,
        "prob_min": float(pred_np.min()),
        "prob_max": float(pred_np.max()),
        "prob_mean": float(pred_np.mean()),
        "foreground_voxels": int(binary_arr.sum()),
        "foreground_ratio": float(binary_arr.mean()),
    }


def prepare_uploaded_volume_for_dataset(
    *,
    uploaded_file: Path,
    destination_directory: Path,
    file_patterns: list[str],
    upload_identifier: str,
) -> Path:
    destination_directory.mkdir(parents=True, exist_ok=True)
    target_name = inference_case_filename(file_patterns, upload_identifier)
    target_path = destination_directory / target_name
    shutil.copyfile(uploaded_file, target_path)
    return target_path
