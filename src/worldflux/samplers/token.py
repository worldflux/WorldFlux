"""Token sampling utilities."""

from __future__ import annotations

import torch
from torch import Tensor


class TokenSampler:
    """Sample discrete tokens from logits."""

    def sample(self, logits: Tensor, temperature: float = 1.0) -> Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        return dist.sample()
