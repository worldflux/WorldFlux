#!/usr/bin/env python3
"""
Train DreamerV3 World Model on Atari data.

This script trains a DreamerV3 world model using trajectories collected
from Atari environments, validating the model on real visual observations.

Usage:
    # Quick test with collected data
    python examples/train_atari_dreamer.py --test

    # Train with real data
    python examples/train_atari_dreamer.py --data atari_data.npz --steps 10000

    # Full training run
    python examples/train_atari_dreamer.py --data atari_data.npz --steps 100000 --wandb

Requirements:
    First collect data with:
        python examples/collect_atari.py

    Then train:
        python examples/train_atari_dreamer.py --data atari_data.npz
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train DreamerV3 on Atari data")
    parser.add_argument(
        "--data",
        type=str,
        default="atari_data.npz",
        help="Path to collected Atari data (.npz file)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with minimal training",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10_000,
        help="Total training steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=50,
        help="Sequence length for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/atari_dreamer",
        help="Output directory for checkpoints",
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
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="Use all available GPUs with DataParallel",
    )
    parser.add_argument(
        "--deter-dim",
        type=int,
        default=512,
        help="Deterministic state dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--cnn-depth",
        type=int,
        default=32,
        help="CNN depth multiplier",
    )
    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {args.data}")
        logger.error("First collect data with: python examples/collect_atari.py")
        sys.exit(1)

    # Import here to give better error messages for missing deps
    from worldflux import create_world_model
    from worldflux.training import ReplayBuffer, Trainer, TrainingConfig
    from worldflux.training.callbacks import (
        EarlyStoppingCallback,
        LoggingCallback,
        ProgressCallback,
    )

    # Try to import collect_atari helper
    try:
        from collect_atari import load_atari_to_buffer
    except ImportError:
        # Define inline if import fails
        def load_atari_to_buffer(path: str, capacity: int | None = None) -> ReplayBuffer:
            """Load collected Atari data into a ReplayBuffer."""
            logger.info(f"Loading data from: {path}")
            data = np.load(path, allow_pickle=False)

            obs = data["obs"]
            actions_onehot = data["actions_onehot"]
            rewards = data["rewards"]
            dones = data["dones"]

            logger.info(f"Loaded {len(obs)} transitions")

            if capacity is None:
                capacity = len(obs)

            obs_shape = obs.shape[1:]
            action_dim = actions_onehot.shape[1]

            buffer = ReplayBuffer(
                capacity=capacity,
                obs_shape=obs_shape,
                action_dim=action_dim,
            )

            # Split into episodes
            episode_starts = [0] + list(np.where(dones[:-1] == 1.0)[0] + 1)
            episode_ends = list(np.where(dones == 1.0)[0] + 1)

            if len(episode_ends) < len(episode_starts):
                episode_ends.append(len(obs))

            num_episodes = min(len(episode_starts), len(episode_ends))

            for i in range(num_episodes):
                start = episode_starts[i]
                end = episode_ends[i]
                if start >= end:
                    continue
                buffer.add_episode(
                    obs=obs[start:end],
                    actions=actions_onehot[start:end],
                    rewards=rewards[start:end],
                    dones=dones[start:end],
                )

            logger.info(f"Buffer: {len(buffer)} transitions, {buffer.num_episodes} episodes")
            return buffer

    # Test mode: minimal training
    if args.test:
        logger.info("Running quick test with minimal training...")
        args.steps = 100
        args.batch_size = 8
        args.seq_len = 20

    # Load data
    buffer = load_atari_to_buffer(args.data)

    # Get observation shape and action dim from buffer
    obs_shape = buffer.obs_shape
    action_dim = buffer.action_dim

    logger.info(f"Obs shape: {obs_shape}, Action dim: {action_dim}")

    # Create model
    logger.info("Creating DreamerV3 model for Atari...")
    model = create_world_model(
        "dreamerv3:size12m",
        obs_shape=obs_shape,
        action_dim=action_dim,
        deter_dim=args.deter_dim,
        hidden_dim=args.hidden_dim,
        cnn_depth=args.cnn_depth,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Multi-GPU support
    if args.multi_gpu:
        import torch

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)
        else:
            logger.warning("--multi-gpu specified but only 1 GPU available")

    # Training config
    config = TrainingConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        log_interval=max(1, args.steps // 100),
        save_interval=max(100, args.steps // 10),
    )

    # Callbacks
    callbacks = [
        ProgressCallback(),
        LoggingCallback(),
    ]
    if not args.test:
        callbacks.append(EarlyStoppingCallback(patience=5000))

    # Train
    trainer = Trainer(model, config, callbacks=callbacks)

    logger.info("Starting training...")
    logger.info(f"  Steps: {args.steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Sequence length: {args.seq_len}")
    logger.info(f"  Learning rate: {args.lr}")

    trained_model = trainer.train(buffer)

    # Save final model (unwrap DataParallel if needed)
    output_path = f"{args.output_dir}/atari_dreamer_final"
    logger.info(f"Saving model to {output_path}...")
    import torch

    if isinstance(trained_model, torch.nn.DataParallel):
        trained_model.module.save_pretrained(output_path)
    else:
        trained_model.save_pretrained(output_path)

    # Print final training metrics
    if hasattr(trainer, "history") and trainer.history:
        logger.info("\nFinal training metrics:")
        for key, values in trainer.history.items():
            if values:
                logger.info(f"  {key}: {values[-1]:.4f}")

    logger.info("Training complete!")

    # Run validation imagination (unwrap DataParallel if needed)
    logger.info("\nRunning validation imagination rollout...")
    model_for_validation = (
        trained_model.module if isinstance(trained_model, torch.nn.DataParallel) else trained_model
    )
    validate_imagination(model_for_validation, buffer, output_dir=args.output_dir)


def validate_imagination(
    model,
    buffer,
    output_dir: str = "./outputs/atari_dreamer",
    horizon: int = 20,
):
    """
    Validate model by running imagination rollout and comparing to real data.

    Args:
        model: Trained DreamerV3 model.
        buffer: ReplayBuffer with real data.
        output_dir: Directory to save visualization.
        horizon: Imagination horizon.
    """
    import torch  # noqa: F811

    model.eval()
    device = next(model.parameters()).device

    # Sample a batch
    batch = buffer.sample(batch_size=1, seq_len=horizon + 1, device=device)
    obs = batch.obs  # [1, T+1, C, H, W]
    actions = batch.actions  # [1, T+1, A]
    rewards = batch.rewards  # [1, T+1]

    # Encode initial observation
    with torch.no_grad():
        initial_obs = obs[:, 0]  # [1, C, H, W]
        state = model.encode(initial_obs)

        # Run imagination with same actions
        action_seq = actions[:, :horizon].permute(1, 0, 2)  # [T, 1, A]
        trajectory = model.rollout(state, action_seq)

        # Decode imagined observations
        imagined_obs = []
        for s in trajectory.states:
            decoded = model.decode(s)
            imagined_obs.append(decoded.preds["obs"].cpu())

    # Compute metrics
    real_obs = obs[:, 1 : horizon + 1].cpu()  # [1, T, C, H, W]
    pred_obs = torch.stack(imagined_obs[1:], dim=1)  # Skip initial state

    # Reconstruction MSE
    mse = torch.mean((real_obs - pred_obs) ** 2).item()
    logger.info(f"Imagination MSE: {mse:.6f}")

    # Reward prediction vs real
    real_rewards = rewards[:, 1 : horizon + 1].cpu()
    pred_rewards = trajectory.rewards.cpu()
    reward_mse = torch.mean((real_rewards - pred_rewards) ** 2).item()
    logger.info(f"Reward MSE: {reward_mse:.6f}")

    # Save visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 6, figsize=(18, 6))

        # Top row: Real observations
        axes[0, 0].set_ylabel("Real", fontsize=12)
        for i in range(6):
            t = i * (horizon // 5)
            if t < real_obs.shape[1]:
                img = real_obs[0, t].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[0, i].imshow(img)
            axes[0, i].set_title(f"t={t}")
            axes[0, i].axis("off")

        # Bottom row: Imagined observations
        axes[1, 0].set_ylabel("Imagined", fontsize=12)
        for i in range(6):
            t = i * (horizon // 5)
            if t < pred_obs.shape[1]:
                img = pred_obs[0, t].permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                axes[1, i].imshow(img)
            axes[1, i].axis("off")

        plt.suptitle(f"Imagination Rollout (MSE: {mse:.4f})", fontsize=14)
        plt.tight_layout()

        output_path = Path(output_dir) / "imagination_rollout.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        logger.info(f"Saved visualization to: {output_path}")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not installed, skipping visualization")


if __name__ == "__main__":
    main()
