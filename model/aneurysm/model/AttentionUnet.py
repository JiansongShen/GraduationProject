from __future__ import annotations

import torch
from torch import nn

from model.aneurysm.model.Unet import Unet
from model.aneurysm.modules.ASPP3D import ASPP3D
from model.aneurysm.modules.CoordAttention3D import CoordAttention3D
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
        use_coord_attention: bool = True,
        coord_reduction: int = 16,
        use_aspp: bool = True,
        aspp_dilations: tuple[int, ...] = (1, 2, 4, 6),
    ):
        super().__init__()
        self.depth = depth

        self.encoder = UnetEncoder(in_ch, base_filter, self.depth, norm_type, activation, dropout)
        bottleneck_ch = self.encoder.channels()[-1]
        self.use_aspp = use_aspp
        self.use_coord_attention = use_coord_attention
        self.aspp = ASPP3D(bottleneck_ch, bottleneck_ch, aspp_dilations) if use_aspp else nn.Identity()
        self.coord_attention = (
            CoordAttention3D(bottleneck_ch, reduction=coord_reduction) if use_coord_attention else nn.Identity()
        )
        self.decoder = UnetDecoder(self.encoder.channels(), out_ch, self.depth, norm_type, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, bottleneck = self.encoder(x)
        bottleneck = self.aspp(bottleneck)
        bottleneck = self.coord_attention(bottleneck)
        return self.decoder(features, bottleneck)

    def get_name(self) -> str:
        return "AttentionUnet"
