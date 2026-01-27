#!/usr/bin/env python3
"""
Example: Training TD-MPC2 World Model

This script demonstrates how to train a TD-MPC2 world model using
either random data (for testing) or data from a replay buffer.

TD-MPC2 is designed for state-based (low-dimensional) observations,
making it suitable for robotics and control tasks.

Usage:
    # Quick test with random data
    python examples/train_tdmpc2.py --test

    # Train with real data
    python examples/train_tdmpc2.py --data trajectories.npz --steps 100000

    # Resume from checkpoint
    python examples/train_tdmpc2.py --data trajectories.npz --resume outputs/checkpoint_best.pt
"""

import argparse
import logging

from worldloom import create_world_model
from worldloom.training import (
    ReplayBuffer,
    Trainer,
    TrainingConfig,
)
from worldloom.training.callbacks import EarlyStoppingCallback, ProgressCallback
from worldloom.training.data import create_random_buffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train TD-MPC2 World Model")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to replay buffer (.npz file)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with random data",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="5m",
        choices=["ci", "5m", "19m", "48m", "317m"],
        help="Model size preset",
    )
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=39,
        help="Observation dimension (for state-based envs)",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=6,
        help="Action dimension",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Total training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (TD-MPC2 typically uses larger batches)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=10,
        help="Sequence length (TD-MPC2 uses shorter sequences)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/tdmpc2",
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb logging",
    )
    args = parser.parse_args()

    # TD-MPC2 uses state-based observations
    obs_shape = (args.obs_dim,)
    action_dim = args.action_dim

    # Load or create data
    if args.test:
        logger.info("Running quick test with random data...")
        args.steps = 10  # Quick test
        args.size = "ci"  # Use tiny model for CI
        buffer = create_random_buffer(
            capacity=10000,
            obs_shape=obs_shape,
            action_dim=action_dim,
            num_episodes=100,
            episode_length=100,
            seed=42,
        )
    elif args.data:
        logger.info(f"Loading data from {args.data}...")
        buffer = ReplayBuffer.load(args.data)
        obs_shape = buffer.obs_shape
        action_dim = buffer.action_dim
    else:
        parser.error("Either --data or --test must be specified")

    logger.info(f"Buffer: {len(buffer)} transitions, {buffer.num_episodes} episodes")
    logger.info(f"Obs shape: {obs_shape}, Action dim: {action_dim}")

    # Create model
    logger.info(f"Creating TD-MPC2 model ({args.size})...")
    model = create_world_model(
        f"tdmpc2:{args.size}",
        obs_shape=obs_shape,
        action_dim=action_dim,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Training config (TD-MPC2 specific defaults)
    config = TrainingConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        log_interval=max(1, args.steps // 100),  # ~100 log lines
        save_interval=max(100, args.steps // 10),  # ~10 checkpoints
    )

    # Callbacks
    callbacks = [ProgressCallback()]
    if not args.test:
        callbacks.append(EarlyStoppingCallback(patience=10000))

    # Train
    trainer = Trainer(model, config, callbacks=callbacks)

    if args.resume:
        logger.info(f"Resuming from {args.resume}...")
        trainer.load_checkpoint(args.resume)

    logger.info("Starting training...")
    trained_model = trainer.train(buffer)

    # Save final model
    output_path = f"{args.output_dir}/tdmpc2_final"
    logger.info(f"Saving model to {output_path}...")
    trained_model.save_pretrained(output_path)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
