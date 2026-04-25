from __future__ import annotations

import torch
from torch import nn

from model.aneurysm.modules.AttentionGate import AttentionGate
from model.aneurysm.modules.ConvBlock3D import ConvBlock3D


class UpSampleBlock(nn.Module):
    """One decoder stage of a 3D Attention U-Net.

    The block performs three steps:
    1. Upsample the decoder tensor.
    2. Gate the corresponding skip tensor with attention.
    3. Concatenate both tensors and refine them with convolution layers.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        skip_ch: int,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attention = AttentionGate(in_ch // 2, skip_ch, in_ch // 2)
        self.conv = ConvBlock3D(in_ch, out_ch, norm_type, activation, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample the decoder tensor and merge it with the attention-filtered skip."""
        x = self.up(x)
        skip = self.attention(x, skip)
        x = torch.cat((x, skip), dim=1)
        return self.conv(x)
