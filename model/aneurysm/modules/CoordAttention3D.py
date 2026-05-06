from __future__ import annotations

import torch
from torch import nn


class CoordAttention3D(nn.Module):
    """Lightweight 3D coordinate attention.

    Encodes long-range dependency along depth/height/width independently,
    then reweights input features with axis-aware attention maps.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(8, channels // max(1, reduction))
        self.shared = nn.Sequential(
            nn.Conv3d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm3d(mid),
            nn.ReLU(inplace=True),
        )
        self.attn_d = nn.Conv3d(mid, channels, kernel_size=1, bias=True)
        self.attn_h = nn.Conv3d(mid, channels, kernel_size=1, bias=True)
        self.attn_w = nn.Conv3d(mid, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_ctx = x.mean(dim=(3, 4), keepdim=True)
        h_ctx = x.mean(dim=(2, 4), keepdim=True)
        w_ctx = x.mean(dim=(2, 3), keepdim=True)

        d_attn = self.sigmoid(self.attn_d(self.shared(d_ctx)))
        h_attn = self.sigmoid(self.attn_h(self.shared(h_ctx)))
        w_attn = self.sigmoid(self.attn_w(self.shared(w_ctx)))
        return x * d_attn * h_attn * w_attn
