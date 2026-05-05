from __future__ import annotations

from dataclasses import dataclass

import SimpleITK as sitk


@dataclass(frozen=True)
class ResampleMeta:
    original_spacing: tuple[float, float, float]
    original_size: tuple[int, int, int]
    original_origin: tuple[float, float, float]
    original_direction: tuple[float, ...]
    train_spacing: tuple[float, float, float]


class ResampleHelper:
    """统一重采样工具：训练空间 <-> 原始NIfTI空间。"""

    def __init__(self, train_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> None:
        self.train_spacing = train_spacing

    def to_train_space(self, image: sitk.Image, *, is_label: bool) -> tuple[sitk.Image, ResampleMeta]:
        meta = ResampleMeta(
            original_spacing=image.GetSpacing(),
            original_size=image.GetSize(),
            original_origin=image.GetOrigin(),
            original_direction=image.GetDirection(),
            train_spacing=self.train_spacing,
        )
        resampled = self._resample(
            image,
            out_spacing=self.train_spacing,
            out_size=self._compute_size(image.GetSize(), image.GetSpacing(), self.train_spacing),
            ref_origin=image.GetOrigin(),
            ref_direction=image.GetDirection(),
            is_label=is_label,
        )
        return resampled, meta

    def to_original_space(self, image_train: sitk.Image, meta: ResampleMeta, *, is_label: bool) -> sitk.Image:
        restored = self._resample(
            image_train,
            out_spacing=meta.original_spacing,
            out_size=meta.original_size,
            ref_origin=meta.original_origin,
            ref_direction=meta.original_direction,
            is_label=is_label,
        )
        return restored

    @staticmethod
    def _compute_size(size: tuple[int, int, int], in_spacing: tuple[float, float, float], out_spacing: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple[int, int, int](tuple(max(1, int(round(size[i] * in_spacing[i] / out_spacing[i]))) for i in range(3)))

    @staticmethod
    def _resample(
        image: sitk.Image,
        *,
        out_spacing: tuple[float, float, float],
        out_size: tuple[int, int, int],
        ref_origin: tuple[float, float, float],
        ref_direction: tuple[float, ...],
        is_label: bool,
    ) -> sitk.Image:
        transform = sitk.Transform()
        interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
        pixel_type = sitk.sitkUInt8 if is_label else sitk.sitkFloat32
        return sitk.Resample(
            image,
            out_size,
            transform,
            interpolator,
            ref_origin,
            out_spacing,
            ref_direction,
            0.0,
            pixel_type,
        )
