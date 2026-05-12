"""Adversarial spatial weight kernel for 3D segmentation losses.

The module learns a voxel-wise weight map from prediction/label disagreement.
It is intentionally lightweight: stacked 3x3x3 convolutions observe local
morphological/topological errors while keeping memory overhead low for
128x128x128 patches.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class AdversarialWeightNet(nn.Module):
    """Learn an adversarial voxel-wise loss weight map for 3D segmentation.

    Args:
        in_channels: Number of input channels. Use ``2`` when concatenating
            ``[prediction, target]`` and ``1`` when using an error/difference map.
        hidden_channels: Width of the lightweight local morphology encoder.
        mode: Input construction mode. ``"concat"`` uses ``[prediction, target]``;
            ``"diff"`` uses ``abs(prediction - target)``.
        temperature: Softmax temperature. Lower values produce sharper weights.
        min_weight: Lower clamp after normalization to avoid zeroing easy voxels.
        max_weight: Upper clamp after normalization to prevent weight explosion.
        normalize: ``"softmax"`` normalizes weights over all spatial voxels and
            rescales them to mean 1; ``"mean"`` uses sigmoid scores divided by
            their mean.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 8,
        mode: str = "concat",
        temperature: float = 1.0,
        min_weight: float = 0.1,
        max_weight: float = 5.0,
        normalize: str = "softmax",
    ) -> None:
        super().__init__()
        if mode not in {"concat", "diff"}:
            raise ValueError(f"Unsupported mode: {mode}. Expected 'concat' or 'diff'.")
        if normalize not in {"softmax", "mean"}:
            raise ValueError(f"Unsupported normalize: {normalize}. Expected 'softmax' or 'mean'.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if min_weight < 0.0 or max_weight <= min_weight:
            raise ValueError("Require 0 <= min_weight < max_weight.")

        expected_channels = 2 if mode == "concat" else 1
        if in_channels != expected_channels:
            raise ValueError(f"mode='{mode}' expects in_channels={expected_channels}, got {in_channels}.")

        self.mode = mode
        self.temperature = temperature
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.normalize = normalize

        self.kernel = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def build_input(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Build the discriminator input from prediction and target tensors."""
        if prediction.shape != target.shape:
            raise ValueError(f"prediction and target must have the same shape, got {prediction.shape} and {target.shape}.")
        if self.mode == "concat":
            return torch.cat([prediction, target], dim=1)
        return (prediction - target).abs()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return a bounded weight map with the same shape as ``prediction``.

        During the discriminator update, pass ``prediction.detach()`` so gradients
        from the adversarial weight kernel do not leak into the segmentation net.
        During the generator update, freeze this module's parameters and normally
        detach the returned weight map before multiplying by BCE.
        """
        features = self.build_input(prediction, target)
        logits = self.kernel(features)

        if self.normalize == "softmax":
            batch_size = logits.shape[0]
            flat_logits = logits.flatten(start_dim=1) / self.temperature
            flat_weights = F.softmax(flat_logits, dim=1) * flat_logits.shape[1]
            weights = flat_weights.view(batch_size, 1, *logits.shape[2:])
        else:
            scores = torch.sigmoid(logits / self.temperature)
            reduce_dims = tuple(range(2, scores.ndim))
            weights = scores / scores.mean(dim=reduce_dims, keepdim=True).clamp_min(1e-6)

        return weights.clamp(min=self.min_weight, max=self.max_weight)


def weighted_bce_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute mean voxel-wise weighted BCE for probability predictions."""
    prediction = prediction.clamp(min=eps, max=1.0 - eps)
    bce_map = F.binary_cross_entropy(prediction, target, reduction="none")
    return (weight * bce_map).mean()
