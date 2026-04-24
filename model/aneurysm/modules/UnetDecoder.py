from typing import Any

import torch
from jupyter_client import channels
from torch import nn

from model.aneurysm.modules.UpSampleBlock import UpSampleBlock
from model.aneurysm.modules.common import _map_final_activation


class UnetDecoder(nn.Module):
    def __init__(self, in_ch: list[int], out_ch: int, depth: int, norm_type: str = "batch", activation: str = "relu",
                 dropout: float = 0.2, final_activation: str = "sigmoid", *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)
        self.depth = depth
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.up_blocks = nn.ModuleList()
        self.final_activation = _map_final_activation(final_activation)

        rev_channels: list[int] = list(reversed(in_ch))

        for i in range(len(rev_channels) - 1):
            self.up_blocks.append(
                UpSampleBlock(
                    rev_channels[i],
                    rev_channels[i + 1],
                    rev_channels[i + 1],
                    norm_type,
                    activation,
                    dropout
                )
            )

        self.out_conv = nn.Conv3d(rev_channels[0], out_ch, kernel_size=1)


    def forward(self, features: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        for i in range(len(features)):
            x = self.up_blocks[i](x, features[len(features) - i - i])

        x = self.out_conv(x)
        return self.final_activation(x)