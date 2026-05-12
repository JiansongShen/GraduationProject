import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size, bias=False, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv(x)
        return x


    def get_weight(self) -> torch.Tensor:
        return self.conv.weight