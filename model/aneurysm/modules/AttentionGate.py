import torch
from jedi.inference.docstrings import infer_param
from torch import nn


class AttentionGate(nn.Module):
    """Attention gate module"""
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_ch, inter_ch, 1),
            nn.BatchNorm3d(inter_ch)
        )
        self.W_s = nn.Sequential(
            nn.Conv3d(skip_ch, inter_ch, 1),
            nn.BatchNorm3d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(inter_ch, 1, 1),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace= True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor):
        W_g = self.W_g(gate)
        W_s = self.W_s(skip)

        psi = self.relu(W_g + W_s)
        psi = self.psi(psi)
        return skip * psi