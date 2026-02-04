#!/usr/bin/env python3
"""
Example: Training a minimal JEPA world model on random data.
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


class RandomJEPABatchProvider:
    """Generate random JEPA batches with context/target/mask."""

    def __init__(self, obs_dim: int):
        self.obs_dim = obs_dim

    def sample(
        self, batch_size: int, seq_len: int | None = None, device: str | torch.device = "cpu"
    ):
        context = torch.randn(batch_size, self.obs_dim, device=device)
        target = torch.randn(batch_size, self.obs_dim, device=device)
        mask = torch.ones(batch_size, 1, device=device)
        return Batch(obs=context, context=context, target=target, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal JEPA model")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--obs-dim", type=int, default=16)
    args = parser.parse_args()

    model = create_world_model(
        "jepa:base",
        obs_shape=(args.obs_dim,),
        action_dim=1,
        encoder_dim=64,
        predictor_dim=64,
    )

    config = TrainingConfig(total_steps=args.steps, batch_size=args.batch_size, sequence_length=1)
    trainer = Trainer(model, config)
    provider = RandomJEPABatchProvider(args.obs_dim)
    trainer.train(provider)
    logger.info("JEPA training complete")


if __name__ == "__main__":
    main()
