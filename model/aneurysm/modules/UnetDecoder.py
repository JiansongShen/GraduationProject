from __future__ import annotations

from typing import Any

import torch
from torch import nn

from model.aneurysm.modules.UpSampleBlock import UpSampleBlock
from model.aneurysm.modules.common import _map_final_activation


class UnetDecoder(nn.Module):
    """3D U-Net decoder.

    The decoder mirrors the encoder and progressively restores the spatial
    resolution using transpose convolutions and attention-gated skip connections.
    """

    def __init__(
        self,
        in_ch: list[int],
        out_ch: int,
        depth: int,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.2,
        final_activation: str = "sigmoid",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        if len(in_ch) < 2:
            raise ValueError("`in_ch` must contain at least two channel sizes.")

        self.depth = depth
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.up_blocks = nn.ModuleList()
        self.final_activation = _map_final_activation(final_activation)

        # Reverse the encoder channels so we can walk from the bottleneck back to
        # the highest-resolution skip feature.
        rev_channels: list[int] = list(reversed(in_ch))

        # Each up block receives the current decoder feature map, upsamples it,
        # merges it with the corresponding skip tensor, and refines the result.
        for i in range(len(rev_channels) - 1):
            current_ch = rev_channels[i]
            next_ch = rev_channels[i + 1]
            self.up_blocks.append(
                UpSampleBlock(
                    current_ch,
                    next_ch,
                    next_ch,
                    norm_type,
                    activation,
                    dropout,
                )
            )

        # After the final decoder stage, map features to the desired output class
        # count using a 1x1x1 convolution.
        self.out_conv = nn.Conv3d(rev_channels[-1], out_ch, kernel_size=1)

    def forward(self, features: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        """Decode the bottleneck using skip features from the encoder.

        Parameters
        ----------
        features:
            Skip tensors ordered from shallow to deep.
        x:
            Bottleneck tensor produced by the encoder.
        """
        for i, block in enumerate(self.up_blocks):
            skip = features[-(i + 1)]
            x = block(x, skip)

        x = self.out_conv(x)
        return self.final_activation(x)
