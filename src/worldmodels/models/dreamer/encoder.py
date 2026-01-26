"""Encoder modules for DreamerV3."""

import torch
import torch.nn as nn
from torch import Tensor


class CNNEncoder(nn.Module):
    """CNN encoder for image observations."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        depth: int = 48,
        kernels: tuple[int, ...] = (4, 4, 4, 4),
        stride: int = 2,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.depth = depth

        in_channels = obs_shape[0]
        layers = []

        for i, kernel in enumerate(kernels):
            out_channels = depth * (2**i)
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel // 2 - 1),
                    nn.LayerNorm(
                        [
                            out_channels,
                            obs_shape[1] // (stride ** (i + 1)),
                            obs_shape[2] // (stride ** (i + 1)),
                        ]
                    ),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Calculate output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            out = self.conv(dummy)
            self._output_dim = out.flatten(1).shape[1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, obs: Tensor) -> Tensor:
        x = self.conv(obs)
        return x.flatten(start_dim=1)


class MLPEncoder(nn.Module):
    """MLP encoder for vector observations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        self._output_dim = output_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, obs: Tensor) -> Tensor:
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        return self.mlp(obs)
