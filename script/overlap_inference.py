from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger("gradulate.split_inference")


@dataclass
class OverlapInferenceConfig:
    """split 推理配置。
    每个 patch 仅保留中心 effective_size 区域拼接，边界不足统一视为 0 padding。
    """

    enabled: bool = True
    patch_size: tuple[int, int, int] = (64, 64, 64)
    effective_size: tuple[int, int, int] = (48, 48, 48)
    batch_size: int = 4
    use_amp: bool = True

    @classmethod
    def from_dict(cls, data: dict | None) -> "OverlapInferenceConfig":
        if not data:
            return cls()
        kwargs = dict(data)
        for key in ("patch_size", "effective_size"):
            if key in kwargs and isinstance(kwargs[key], list):
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)


def _center_offsets(patch_size: tuple[int, int, int], effective_size: tuple[int, int, int]) -> tuple[int, int, int]:
    offsets = []
    for patch_dim, eff_dim in zip(patch_size, effective_size):
        if eff_dim <= 0 or patch_dim <= 0 or eff_dim > patch_dim:
            raise ValueError(f"Invalid patch/effective size: patch_size={patch_size}, effective_size={effective_size}")
        offsets.append((patch_dim - eff_dim) // 2)
    return tuple(offsets)


def predict_with_overlap(
    model: torch.nn.Module,
    image: torch.Tensor,
    config: OverlapInferenceConfig,
    device: torch.device,
    show_progress: bool = False,
) -> torch.Tensor:
    """保留旧函数名，内部实现为中心有效区域拼接。"""
    _ = show_progress
    model.eval()

    if image.ndim == 5:
        volume = image.squeeze(0).squeeze(0)
    elif image.ndim == 3:
        volume = image
    else:
        raise ValueError(f"Unsupported image shape for split inference: {tuple(image.shape)}")

    src_shape = tuple(int(v) for v in volume.shape)
    patch_size = tuple(int(v) for v in config.patch_size)
    effective_size = tuple(int(v) for v in config.effective_size)
    offsets = _center_offsets(patch_size, effective_size)

    prediction = torch.zeros(src_shape, dtype=torch.float32)
    stride = effective_size
    patch_batch: list[torch.Tensor] = []
    batch_positions: list[tuple[int, int, int]] = []

    def flush_batch() -> None:
        nonlocal patch_batch, batch_positions, prediction
        if not patch_batch:
            return
        batch_tensor = torch.stack(patch_batch, dim=0).to(device)
        with torch.no_grad():
            if config.use_amp and device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(batch_tensor)
            else:
                outputs = model(batch_tensor)
        outputs = outputs.detach().cpu()
        for pred_patch, start in zip(outputs, batch_positions):
            while pred_patch.ndim > 3:
                pred_patch = pred_patch[0]
            center_pred = pred_patch[
                offsets[0]: offsets[0] + effective_size[0],
                offsets[1]: offsets[1] + effective_size[1],
                offsets[2]: offsets[2] + effective_size[2],
            ].clamp(0.0, 1.0)
            z, y, x = start
            z_end = min(z + effective_size[0], src_shape[0])
            y_end = min(y + effective_size[1], src_shape[1])
            x_end = min(x + effective_size[2], src_shape[2])
            prediction[z:z_end, y:y_end, x:x_end] = center_pred[: z_end - z, : y_end - y, : x_end - x]
        patch_batch = []
        batch_positions = []

    for z in range(0, src_shape[0], stride[0]):
        for y in range(0, src_shape[1], stride[1]):
            for x in range(0, src_shape[2], stride[2]):
                patch_start = (z - offsets[0], y - offsets[1], x - offsets[2])
                patch = torch.zeros(patch_size, dtype=torch.float32)
                src_z0 = max(0, patch_start[0])
                src_y0 = max(0, patch_start[1])
                src_x0 = max(0, patch_start[2])
                src_z1 = min(src_shape[0], patch_start[0] + patch_size[0])
                src_y1 = min(src_shape[1], patch_start[1] + patch_size[1])
                src_x1 = min(src_shape[2], patch_start[2] + patch_size[2])
                dst_z0 = src_z0 - patch_start[0]
                dst_y0 = src_y0 - patch_start[1]
                dst_x0 = src_x0 - patch_start[2]
                patch[
                    dst_z0:dst_z0 + (src_z1 - src_z0),
                    dst_y0:dst_y0 + (src_y1 - src_y0),
                    dst_x0:dst_x0 + (src_x1 - src_x0),
                ] = volume[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
                patch_batch.append(patch.unsqueeze(0))
                batch_positions.append((z, y, x))
                if len(patch_batch) >= max(1, int(config.batch_size)):
                    flush_batch()

    flush_batch()
    logger.info(
        "Split inference completed with center stitching. src_shape=%s patch_size=%s effective_size=%s",
        src_shape,
        patch_size,
        effective_size,
    )
    return prediction.unsqueeze(0).unsqueeze(0)


OverlapInferenceConfigProtocol = OverlapInferenceConfig
