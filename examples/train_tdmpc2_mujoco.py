#!/usr/bin/env python3
"""
Train TD-MPC2 World Model on MuJoCo continuous control data.

This script trains a TD-MPC2 world model using trajectories collected
from MuJoCo environments, validating the model on continuous control tasks.

Usage:
    # Quick test with collected data
    python examples/train_tdmpc2_mujoco.py --test

    # Train with real data
    python examples/train_tdmpc2_mujoco.py --data mujoco_data.npz --steps 10000

Requirements:
    First collect data with:
        python examples/collect_mujoco.py

    Then train:
        python examples/train_tdmpc2_mujoco.py --data mujoco_data.npz
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train TD-MPC2 on MuJoCo data")
    parser.add_argument(
        "--data",
        type=str,
        default="mujoco_data.npz",
        help="Path to collected MuJoCo data (.npz file)",
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
        default=32,
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
        default="./outputs/tdmpc2_mujoco",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent state dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for TD learning",
    )
    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {args.data}")
        logger.error("First collect data with: python examples/collect_mujoco.py")
        sys.exit(1)

    # Import here to give better error messages for missing deps
    from worldflux import create_world_model
    from worldflux.training import ReplayBuffer, Trainer, TrainingConfig
    from worldflux.training.callbacks import (
        EarlyStoppingCallback,
        LoggingCallback,
        ProgressCallback,
    )

    # Try to import collect_mujoco helper
    try:
        from collect_mujoco import load_mujoco_to_buffer
    except ImportError:
        # Define inline if import fails
        def load_mujoco_to_buffer(path: str, capacity: int | None = None) -> ReplayBuffer:
            """Load collected MuJoCo data into a ReplayBuffer."""
            logger.info(f"Loading data from: {path}")
            data = np.load(path, allow_pickle=False)

            obs = data["obs"]
            actions = data["actions"]
            rewards = data["rewards"]
            dones = data["dones"]
            obs_dim = int(data["obs_dim"])
            action_dim = int(data["action_dim"])

            logger.info(f"Loaded {len(obs)} transitions")

            if capacity is None:
                capacity = len(obs)

            buffer = ReplayBuffer(
                capacity=capacity,
                obs_shape=(obs_dim,),
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
                    actions=actions[start:end],
                    rewards=rewards[start:end],
                    dones=dones[start:end],
                )

            logger.info(f"Buffer: {len(buffer)} transitions, {buffer.num_episodes} episodes")
            return buffer

    # Test mode: minimal training
    if args.test:
        logger.info("Running quick test with minimal training...")
        args.steps = 100
        args.batch_size = 16
        args.seq_len = 20

    # Load data
    buffer = load_mujoco_to_buffer(args.data)

    # Get observation shape and action dim from buffer
    obs_shape = buffer.obs_shape
    action_dim = buffer.action_dim

    logger.info(f"Obs shape: {obs_shape}, Action dim: {action_dim}")

    # Create TD-MPC2 model
    logger.info("Creating TD-MPC2 model...")
    model = create_world_model(
        "tdmpc2:5m",
        obs_shape=obs_shape,
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

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
    logger.info(f"  Gamma: {args.gamma}")

    trained_model = trainer.train(buffer)

    # Save final model
    output_path = f"{args.output_dir}/tdmpc2_final"
    logger.info(f"Saving model to {output_path}...")
    trained_model.save_pretrained(output_path)

    # Print final training metrics
    if hasattr(trainer, "history") and trainer.history:
        logger.info("\nFinal training metrics:")
        for key, values in trainer.history.items():
            if values:
                logger.info(f"  {key}: {values[-1]:.4f}")

    logger.info("Training complete!")

    # Run validation
    logger.info("\nRunning validation...")
    validate_tdmpc2(trained_model, buffer, output_dir=args.output_dir)


def validate_tdmpc2(
    model,
    buffer,
    output_dir: str = "./outputs/tdmpc2_mujoco",
    horizon: int = 20,
):
    """
    Validate TD-MPC2 model by checking latent predictions and Q-values.

    Args:
        model: Trained TD-MPC2 model.
        buffer: ReplayBuffer with real data.
        output_dir: Directory to save visualization.
        horizon: Prediction horizon.
    """
    model.eval()
    device = next(model.parameters()).device

    # Sample a batch
    batch = buffer.sample(batch_size=1, seq_len=horizon + 1, device=device)
    obs = batch.obs  # [1, T+1, obs_dim]
    actions = batch.actions  # [1, T+1, action_dim]
    rewards = batch.rewards  # [1, T+1]

    with torch.no_grad():
        # Encode initial observation
        initial_obs = obs[:, 0]  # [1, obs_dim]
        state = model.encode(initial_obs)

        # Run imagination with same actions
        action_seq = actions[:, :horizon].permute(1, 0, 2)  # [T, 1, action_dim]
        trajectory = model.rollout(state, action_seq)

        # Get Q-values for initial state
        q_values = model.predict_q(state, actions[:, 0])
        logger.info(f"Initial Q-values (ensemble): {q_values.squeeze().cpu().numpy()}")
        logger.info(f"Q-value mean: {q_values.mean().item():.4f}")

    # Compute metrics
    real_rewards = rewards[:, 1 : horizon + 1].cpu()  # [1, T]
    pred_rewards = trajectory.rewards.cpu()  # [T, 1]

    # Reward prediction MSE
    reward_mse = torch.mean((real_rewards - pred_rewards.T) ** 2).item()
    logger.info(f"Reward prediction MSE: {reward_mse:.6f}")

    # Reward correlation
    real_flat = real_rewards.flatten().numpy()
    pred_flat = pred_rewards.flatten().numpy()
    if len(real_flat) > 1:
        correlation = np.corrcoef(real_flat, pred_flat)[0, 1]
        logger.info(f"Reward correlation: {correlation:.4f}")

    # Latent state consistency
    logger.info("\nLatent state analysis:")
    latent_norms = []
    for i, s in enumerate(trajectory.states[:5]):  # First 5 states
        latent = s.tensors.get("latent")
        if latent is None:
            latent = s.tensors.get("deter")
        if latent is None:
            latent = next(iter(s.tensors.values()), None)
        if latent is not None:
            norm = latent.norm().item()
            latent_norms.append(norm)
            logger.info(f"  State {i} latent norm: {norm:.4f}")

    # Save visualization
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Reward prediction over time
        t = np.arange(horizon)
        axes[0, 0].plot(t, real_flat, "b-", label="Real", linewidth=2)
        axes[0, 0].plot(t, pred_flat, "r--", label="Predicted", linewidth=2)
        axes[0, 0].set_xlabel("Time Step")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].set_title(f"Reward Prediction (MSE: {reward_mse:.4f})")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Reward scatter
        axes[0, 1].scatter(real_flat, pred_flat, alpha=0.6, s=50)
        lims = [min(real_flat.min(), pred_flat.min()), max(real_flat.max(), pred_flat.max())]
        axes[0, 1].plot(lims, lims, "r--", linewidth=2, label="Perfect")
        axes[0, 1].set_xlabel("Real Reward")
        axes[0, 1].set_ylabel("Predicted Reward")
        axes[0, 1].set_title("Predicted vs Real Reward")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Q-values
        q_vals = q_values.squeeze().cpu().numpy()
        axes[1, 0].bar(range(len(q_vals)), q_vals, color="steelblue")
        axes[1, 0].axhline(
            y=q_vals.mean(), color="red", linestyle="--", label=f"Mean: {q_vals.mean():.2f}"
        )
        axes[1, 0].set_xlabel("Q-Network Index")
        axes[1, 0].set_ylabel("Q-Value")
        axes[1, 0].set_title("Q-Value Ensemble")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Latent norm over time
        if latent_norms:
            axes[1, 1].plot(
                range(len(latent_norms)), latent_norms, "g-o", linewidth=2, markersize=8
            )
            axes[1, 1].set_xlabel("Time Step")
            axes[1, 1].set_ylabel("Latent Norm")
            axes[1, 1].set_title("Latent State Stability")
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("TD-MPC2 Validation Results", fontsize=14)
        plt.tight_layout()

        output_path = Path(output_dir) / "tdmpc2_validation.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to: {output_path}")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not installed, skipping visualization")


if __name__ == "__main__":
    main()
