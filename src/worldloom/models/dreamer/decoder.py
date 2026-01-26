"""Decoder modules for DreamerV3."""

import torch.nn as nn
from torch import Tensor


class CNNDecoder(nn.Module):
    """CNN decoder for image reconstruction."""

    def __init__(
        self,
        feature_dim: int,
        obs_shape: tuple[int, ...],
        depth: int = 48,
        kernels: tuple[int, ...] = (4, 4, 4, 4),
        stride: int = 2,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.depth = depth

        # Calculate initial spatial size
        num_layers = len(kernels)
        self.init_h = obs_shape[1] // (stride**num_layers)
        self.init_w = obs_shape[2] // (stride**num_layers)
        self.init_channels = depth * (2 ** (num_layers - 1))

        # Project features to spatial
        self.fc = nn.Linear(feature_dim, self.init_channels * self.init_h * self.init_w)

        # Transpose convolutions
        layers: list[nn.Module] = []
        in_channels = self.init_channels

        for i, kernel in enumerate(reversed(kernels)):
            if i < num_layers - 1:
                out_channels = depth * (2 ** (num_layers - 2 - i))
            else:
                out_channels = obs_shape[0]

            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding=kernel // 2 - 1,
                    output_padding=0,
                )
            )

            if i < num_layers - 1:
                layers.extend(
                    [
                        nn.LayerNorm(
                            [
                                out_channels,
                                self.init_h * (stride ** (i + 1)),
                                self.init_w * (stride ** (i + 1)),
                            ]
                        ),
                        nn.SiLU(),
                    ]
                )

            in_channels = out_channels

        self.deconv = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        x = self.fc(features)
        x = x.view(-1, self.init_channels, self.init_h, self.init_w)
        x = self.deconv(x)
        return x


class MLPDecoder(nn.Module):
    """MLP decoder for vector observations."""

    def __init__(
        self,
        feature_dim: int,
        output_shape: tuple[int, ...],
        hidden_dim: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        self.output_shape = output_shape
        output_dim = 1
        for d in output_shape:
            output_dim *= d

        layers = []
        in_dim = feature_dim
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

    def forward(self, features: Tensor) -> Tensor:
        x = self.mlp(features)
        return x.view(-1, *self.output_shape)
