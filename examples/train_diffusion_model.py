#!/usr/bin/env python3
"""
Example: Training a minimal diffusion world model on random data.
"""

import argparse
import logging

import torch

from worldflux import create_world_model
from worldflux.core.batch import Batch
from worldflux.training import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RandomDiffusionBatchProvider:
    """Generate random diffusion batches."""

    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def sample(
        self, batch_size: int, seq_len: int | None = None, device: str | torch.device = "cpu"
    ):
        obs = torch.randn(batch_size, self.obs_dim, device=device)
        target = torch.randn(batch_size, self.obs_dim, device=device)
        actions = torch.randn(batch_size, self.action_dim, device=device)
        return Batch(obs=obs, actions=actions, target=target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal diffusion world model")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=2)
    args = parser.parse_args()

    model = create_world_model(
        "diffusion:base",
        obs_shape=(args.obs_dim,),
        action_dim=args.action_dim,
        hidden_dim=64,
        diffusion_steps=2,
    )

    config = TrainingConfig(total_steps=args.steps, batch_size=args.batch_size, sequence_length=1)
    trainer = Trainer(model, config)
    provider = RandomDiffusionBatchProvider(args.obs_dim, args.action_dim)
    trainer.train(provider)
    logger.info("Diffusion model training complete")


if __name__ == "__main__":
    main()
