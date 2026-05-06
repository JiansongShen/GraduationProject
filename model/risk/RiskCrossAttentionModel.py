from __future__ import annotations

import math

import torch
from torch import nn

from model.risk.CrossAttentionBlock import CrossAttentionBlock


class RiskCrossAttentionModel(nn.Module):
    """Cross-attention model with separate numeric/categorical token streams."""

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: list[int],
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("`hidden_dim` must be divisible by `num_heads`.")
        if num_numeric_features <= 0:
            raise ValueError("`num_numeric_features` must be positive.")

        self.num_numeric_features = num_numeric_features
        self.num_categorical_features = len(categorical_cardinalities)
        self.hidden_dim = hidden_dim

        self.numeric_scale = nn.Parameter(torch.ones(num_numeric_features))
        self.numeric_token_bias = nn.Parameter(torch.zeros(num_numeric_features, hidden_dim))
        self.numeric_projection = nn.Linear(1, hidden_dim)

        self.categorical_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, hidden_dim) for cardinality in categorical_cardinalities]
        )
        self.categorical_proj = nn.Linear(hidden_dim, hidden_dim) if self.num_categorical_features > 0 else nn.Identity()

        self.numeric_to_categorical = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.categorical_to_numeric = nn.ModuleList(
            [CrossAttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _build_numeric_tokens(self, numeric_x: torch.Tensor) -> torch.Tensor:
        batch_size = numeric_x.size(0)
        scaled = numeric_x * self.numeric_scale
        tokens = self.numeric_projection(scaled.reshape(-1, 1)).reshape(batch_size, self.num_numeric_features, self.hidden_dim)
        return tokens + self.numeric_token_bias.unsqueeze(0)

    def _build_categorical_tokens(self, categorical_x: torch.Tensor) -> torch.Tensor:
        if self.num_categorical_features == 0:
            return torch.empty(categorical_x.size(0), 0, self.hidden_dim, device=categorical_x.device)

        embedded_tokens: list[torch.Tensor] = []
        for idx, embedding in enumerate(self.categorical_embeddings):
            token = embedding(categorical_x[:, idx])
            embedded_tokens.append(token.unsqueeze(1))
        cat_tokens = torch.cat(embedded_tokens, dim=1)
        return self.categorical_proj(cat_tokens)

    def forward_logits(self, numeric_x: torch.Tensor, categorical_x: torch.Tensor) -> torch.Tensor:
        numeric_tokens = self._build_numeric_tokens(numeric_x)
        categorical_tokens = self._build_categorical_tokens(categorical_x)

        if categorical_tokens.size(1) == 0:
            # Fall back to self-conditioning when no categorical feature is selected.
            categorical_tokens = numeric_tokens.mean(dim=1, keepdim=True)

        for num_to_cat, cat_to_num in zip(self.numeric_to_categorical, self.categorical_to_numeric):
            numeric_tokens = num_to_cat(numeric_tokens, categorical_tokens)
            categorical_tokens = cat_to_num(categorical_tokens, numeric_tokens)

        pooled_numeric = numeric_tokens.mean(dim=1)
        pooled_categorical = categorical_tokens.mean(dim=1)
        fused = torch.cat((pooled_numeric, pooled_categorical), dim=1)
        return self.output_head(fused).squeeze(1)

    def forward(self, numeric_x: torch.Tensor, categorical_x: torch.Tensor) -> torch.Tensor:
        """Return risk probability in [0, 1]."""
        return torch.sigmoid(self.forward_logits(numeric_x, categorical_x))

    @staticmethod
    def parameter_count(model: nn.Module) -> int:
        return sum(math.prod(p.shape) for p in model.parameters())
