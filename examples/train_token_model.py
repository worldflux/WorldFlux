#!/usr/bin/env python3
"""
Example: Training a minimal token world model on random token data.
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


class RandomTokenBatchProvider:
    """Generate random token batches for training."""

    def __init__(self, vocab_size: int, seq_len: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def sample(
        self, batch_size: int, seq_len: int | None = None, device: str | torch.device = "cpu"
    ):
        length = seq_len or self.seq_len
        tokens = torch.randint(self.vocab_size, (batch_size, length), device=device)
        target = torch.randint(self.vocab_size, (batch_size, length), device=device)
        return Batch(obs=tokens, target=target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a minimal token world model")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    args = parser.parse_args()

    model = create_world_model(
        "token:base",
        obs_shape=(args.seq_len,),
        action_dim=1,
        vocab_size=args.vocab_size,
        token_dim=64,
        num_layers=2,
        num_heads=2,
    )

    config = TrainingConfig(
        total_steps=args.steps, batch_size=args.batch_size, sequence_length=args.seq_len
    )
    trainer = Trainer(model, config)
    provider = RandomTokenBatchProvider(args.vocab_size, args.seq_len)
    trainer.train(provider)
    logger.info("Token model training complete")


if __name__ == "__main__":
    main()
