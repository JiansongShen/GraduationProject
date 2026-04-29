from __future__ import annotations

import torch
from torch import nn

from model.aneurysm.model.Unet import Unet
from model.aneurysm.modules.UnetDecoder import UnetDecoder
from model.aneurysm.modules.UnetEncoder import UnetEncoder


class AttentionUnet(Unet):
    """Attention U-Net for 3D medical image segmentation.

    The original code mixed up the encoder channel configuration and tried to
    call a list as if it were a function. This implementation keeps the encoder
    and decoder channel bookkeeping explicit and stable.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int = 4,
        base_filter: int = 8,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.depth = depth

        self.encoder = UnetEncoder(in_ch, base_filter, self.depth, norm_type, activation, dropout)
        self.decoder = UnetDecoder(self.encoder.channels(), out_ch, self.depth, norm_type, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, bottleneck = self.encoder(x)
        return self.decoder(features, bottleneck)

    def get_name(self) -> str:
        return "AttentionUnet"
