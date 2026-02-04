"""Diffusion-style sampler stub."""

from __future__ import annotations

from torch import Tensor


class DiffusionSampler:
    """Minimal diffusion sampler for step-wise inference."""

    def step(self, model, x: Tensor, action: Tensor | None = None) -> Tensor:
        return model.denoise(x, action)

    def sample(self, model, x: Tensor, action: Tensor | None = None, steps: int = 1) -> Tensor:
        out = x
        for _ in range(max(1, steps)):
            out = self.step(model, out, action)
        return out
