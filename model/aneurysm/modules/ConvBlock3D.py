import torch
from torch import nn

from model.aneurysm.modules.common import _map_norm_layer, _map_layer_activation


class ConvBlock3D(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, norm_type: str = "batch", activation: str = "relu",
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = _map_norm_layer(norm_type, out_ch)
        self.norm2 = _map_norm_layer(norm_type, out_ch)
        self.activation = _map_layer_activation(activation)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        return x
