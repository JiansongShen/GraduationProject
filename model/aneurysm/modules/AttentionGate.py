from __future__ import annotations

import torch
from torch import nn


class AttentionGate(nn.Module):
    """Attention gate module for skip-connection filtering.

    The gate learns to suppress irrelevant skip features before they are fused
    with the decoder stream. This is particularly useful in medical segmentation,
    where background often dominates the volume.
    """

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_ch, inter_ch, 1),
            nn.BatchNorm3d(inter_ch),
        )
        self.W_s = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, 1),
            nn.BatchNorm3d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor):
        """Return the attention-weighted skip tensor."""
        w_g = self.W_g(gate)
        w_s = self.W_s(skip)

        psi = self.relu(w_g + w_s)
        psi = self.psi(psi)
        return skip * psi
