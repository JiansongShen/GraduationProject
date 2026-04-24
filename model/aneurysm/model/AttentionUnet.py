import torch
from torch import nn

from model.aneurysm.model.Unet import Unet
from model.aneurysm.modules.UnetDecoder import UnetDecoder
from model.aneurysm.modules.UnetEncoder import UnetEncoder
from model.aneurysm.modules.UpSampleBlock import UpSampleBlock
from model.aneurysm.modules.ConvBlock3D import ConvBlock3D


class AttentionUnet(Unet):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 depth: int = 4,
                 norm_type: str = "batch",
                 activation: str = "relu",
                 dropout: float = 0.2):
        super().__init__()
        self.depth = depth
        self.encoder = UnetEncoder(in_ch, out_ch, self.depth, norm_type, activation, dropout)
        self.decoder = UnetDecoder(self.encoder.channels(), out_ch, self.depth, norm_type, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, bottleneck = self.encoder(x)
        return self.decoder(features, bottleneck)


    def get_name(self)-> str:
        return "AttentionUnet"
