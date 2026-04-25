from __future__ import annotations

import torch
from torch import nn

from model.aneurysm.modules.ConvBlock3D import ConvBlock3D


class UnetEncoder(nn.Module):
    """3D U-Net encoder.

    The encoder progressively reduces spatial resolution while increasing the
    number of feature channels. We keep an explicit list of channel sizes so the
    decoder can build skip-connections with the correct tensor shapes.
    """

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        depth: int,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("`depth` must be at least 2 for a U-Net encoder/decoder.")

        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.in_ch = in_ch
        self.base_ch = base_ch
        self.depth = depth

        # First convolution stage.
        self.in_conv = ConvBlock3D(in_ch, base_ch, norm_type, activation, dropout)

        # `channel_sizes` stores the number of channels produced at each encoder
        # stage. The decoder will consume the reversed list to build matching
        # skip connections.
        self.channel_sizes: list[int] = [base_ch]
        for _ in range(depth - 1):
            next_ch = self.channel_sizes[-1] * 2
            self.blocks.append(
                ConvBlock3D(
                    self.channel_sizes[-1],
                    next_ch,
                    norm_type,
                    activation,
                    dropout,
                )
            )
            self.channel_sizes.append(next_ch)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Return skip features and the bottleneck tensor.

        The returned `features` list contains all encoder outputs except the
        bottleneck. The last tensor is the bottleneck feature map that gets fed
        into the decoder.
        """
        x = self.in_conv(x)
        features = [x]
        for block in self.blocks:
            x = self.pool(x)
            x = block(x)
            features.append(x)
        return features[:-1], features[-1]

    def channels(self) -> list[int]:
        """Return encoder channel sizes for decoder construction."""
        return list(self.channel_sizes)
