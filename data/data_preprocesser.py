import logging

import SimpleITK as sitk
from typing import Tuple, Optional, List, Final

NiftiImage = sitk.Image


def resample_in_memory(
        image: NiftiImage,
        target_spacing: Tuple[float, float, float] = (1, 1, 1),
        is_mask: bool = False
) -> NiftiImage:
    """
    Resamples a NIfTI image to a target spacing entirely in RAM.
    """
    # 1. Capture original metadata
    original_spacing: Tuple[float, ...] = image.GetSpacing()
    original_size: Tuple[int, ...] = image.GetSize()

    # 2. Compute new dimensions
    new_size: List[int] = [
        int(round(old_sz * old_sp / new_sp))
        for old_sz, old_sp, new_sp in zip(original_size, original_spacing, target_spacing)
    ]

    # 3. Choose interpolation based on data type
    # NearestNeighbor for masks (0, 1, 2...), Linear for scans (CT/MRI)
    interpolator: Final = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear

    # 4. Setup Resampler
    resampler: sitk.ResampleImageFilter = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    new_img : NiftiImage = resampler.Execute(image)

    # 5. Execute and return the memory-resident object
    logging.debug(f"Resampling {image.GetSize()} to {new_img.GetSize()}...")
    return new_img