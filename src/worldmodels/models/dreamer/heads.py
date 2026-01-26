"""Prediction heads for DreamerV3."""

import torch
import torch.nn as nn
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    """Symlog transform: sign(x) * ln(|x| + 1)"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: Tensor) -> Tensor:
    """Inverse symlog: sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class RewardHead(nn.Module):
    """Reward prediction head."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        use_symlog: bool = True,
    ):
        super().__init__()
        self.use_symlog = use_symlog

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(features)

    def predict(self, features: Tensor) -> Tensor:
        """Get reward prediction (with symexp if needed)."""
        out = self.mlp(features).squeeze(-1)
        if self.use_symlog:
            out = symexp(out)
        return out


class ContinueHead(nn.Module):
    """Continue probability prediction head."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(features)

    def predict(self, features: Tensor) -> Tensor:
        """Get continue probability."""
        return torch.sigmoid(self.mlp(features).squeeze(-1))
