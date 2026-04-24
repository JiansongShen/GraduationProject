import torch
from torch import nn

from model.aneurysm.modules.ConvBlock3D import ConvBlock3D


class UnetEncoder(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 depth: int,
                 norm_type: str = "batch",
                 activation: str = "relu",
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        self.in_ch = in_ch
        self.out_ch = out_ch

        self.in_conv = ConvBlock3D(in_ch, 64, norm_type, activation, dropout)
        self.channels = [out_ch]
        for i in range(depth - 1):
            self.blocks.append(
                ConvBlock3D(self.channels[-1],
                            self.channels[-1] * 2,
                            norm_type,
                            activation,
                            dropout))
            self.channels.append(self.channels[-1] * 2)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        features = [self.in_conv(x)]
        for block in self.blocks:
            features.append(block(features[-1]))
        return features[: -1], features[-1]

    def channels (self):
        return self.channels