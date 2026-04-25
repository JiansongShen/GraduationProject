import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for tabular risk prediction."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Update query tokens by attending to key/value tokens."""
        attn_output, _ = self.attn(query=query, key=key_value, value=key_value, need_weights=False)
        query = self.norm1(query + self.dropout(attn_output))
        ffn_output = self.ffn(query)
        return self.norm2(query + self.dropout(ffn_output))

