from __future__ import annotations

import torch
from torch import nn


class _ASPPBranch3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ASPP3D(nn.Module):
    """3D Atrous Spatial Pyramid Pooling for multi-scale context."""

    def __init__(self, in_ch: int, out_ch: int, dilations: tuple[int, ...] = (1, 2, 4, 6)) -> None:
        super().__init__()
        if len(dilations) == 0:
            raise ValueError("`dilations` for ASPP3D must not be empty.")

        branch_ch = max(8, out_ch // len(dilations))
        self.branches = nn.ModuleList([_ASPPBranch3D(in_ch, branch_ch, d) for d in dilations])
        fuse_in = branch_ch * len(dilations)
        self.project = nn.Sequential(
            nn.Conv3d(fuse_in, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))
