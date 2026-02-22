"""Diffusion sampling utilities with a simple scheduler abstraction."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class DiffusionScheduler:
    """Simple DDPM-style scheduler with linear beta schedule."""

    def __init__(
        self,
        *,
        num_train_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_target: str = "noise",
    ):
        if num_train_steps <= 0:
            raise ValueError(f"num_train_steps must be positive, got {num_train_steps}")
        if beta_start <= 0 or beta_end <= 0 or beta_end <= beta_start:
            raise ValueError(f"Invalid beta range: beta_start={beta_start}, beta_end={beta_end}")
        if prediction_target not in {"noise", "x0"}:
            raise ValueError(f"prediction_target must be 'noise' or 'x0', got {prediction_target}")

        self.num_train_steps = num_train_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_target = prediction_target

        self.betas = torch.linspace(beta_start, beta_end, num_train_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_timesteps(self, batch_size: int, *, device: torch.device) -> Tensor:
        return torch.randint(
            0, self.num_train_steps, (batch_size,), device=device, dtype=torch.long
        )

    def _validate_timesteps(self, timesteps: Tensor) -> None:
        if timesteps.dtype != torch.long:
            raise ValueError(f"timesteps must be torch.long, got {timesteps.dtype}")
        if timesteps.numel() == 0:
            return
        min_t = int(timesteps.min().item())
        max_t = int(timesteps.max().item())
        if min_t < 0 or max_t >= self.num_train_steps:
            raise ValueError(
                f"timesteps out of range [0, {self.num_train_steps - 1}], "
                f"got min={min_t}, max={max_t}"
            )

    def add_noise(self, clean: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        self._validate_timesteps(timesteps)
        alpha_bar = self._extract(self.alpha_cumprod, timesteps, clean)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

    def step(self, prediction: Tensor, x_t: Tensor, timesteps: Tensor) -> Tensor:
        self._validate_timesteps(timesteps)
        alpha_bar = self._extract(self.alpha_cumprod, timesteps, x_t)
        if self.prediction_target == "x0":
            x0 = prediction
        else:
            x0 = (x_t - (1.0 - alpha_bar).sqrt() * prediction) / alpha_bar.sqrt().clamp_min(1e-8)
        return x0

    @staticmethod
    def _extract(values: Tensor, timesteps: Tensor, ref: Tensor) -> Tensor:
        gathered = values.to(device=ref.device, dtype=ref.dtype)[timesteps]
        while gathered.dim() < ref.dim():
            gathered = gathered.unsqueeze(-1)
        return gathered


class DiffusionSampler:
    """Diffusion sampler using model denoiser + scheduler."""

    def __init__(self, scheduler: DiffusionScheduler | None = None):
        self.scheduler = scheduler or DiffusionScheduler()

    def step(
        self,
        model: Any,
        x: Tensor,
        action: Tensor | None = None,
        timesteps: Tensor | None = None,
    ) -> Tensor:
        if timesteps is None:
            timesteps = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        try:
            prediction = model.denoise(x, action, timestep=timesteps)
        except TypeError:
            prediction = model.denoise(x, action)
        return self.scheduler.step(prediction, x, timesteps)

    def sample(
        self,
        model: Any,
        x: Tensor,
        action: Tensor | None = None,
        steps: int = 1,
        start_timestep: int | None = None,
    ) -> Tensor:
        out = x
        total_steps = max(1, steps)
        if start_timestep is None:
            start_timestep = self.scheduler.num_train_steps - 1
        for i in range(total_steps):
            t = max(start_timestep - i, 0)
            timesteps = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
            out = self.step(model, out, action, timesteps=timesteps)
        return out
