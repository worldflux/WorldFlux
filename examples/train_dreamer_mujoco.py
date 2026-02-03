#!/usr/bin/env python3
"""
Train DreamerV3 World Model on MuJoCo continuous control data.

This validates DreamerV3 on state-based (non-visual) observations,
demonstrating its flexibility beyond image inputs.

Usage:
    python examples/train_dreamer_mujoco.py --data mujoco_data.npz --test
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train DreamerV3 on MuJoCo data")
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
        default=5000,
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
        "--output-dir",
        type=str,
        default="./outputs/dreamer_mujoco",
        help="Output directory for checkpoints",
    )
    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {args.data}")
        logger.error("First collect data with: python examples/collect_mujoco.py")
        sys.exit(1)

    # Import WorldFlux
    from worldflux import create_world_model
    from worldflux.training import ReplayBuffer, Trainer, TrainingConfig
    from worldflux.training.callbacks import (
        EarlyStoppingCallback,
        LoggingCallback,
        ProgressCallback,
    )

    # Test mode
    if args.test:
        logger.info("Running quick test with minimal training...")
        args.steps = 100
        args.batch_size = 16
        args.seq_len = 20

    # Load data
    logger.info(f"Loading data from: {args.data}")
    data = np.load(args.data, allow_pickle=False)

    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    obs_dim = int(data["obs_dim"])
    action_dim = int(data["action_dim"])

    logger.info(f"Loaded {len(obs)} transitions")
    logger.info(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Create buffer
    buffer = ReplayBuffer(
        capacity=len(obs),
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

    # Create DreamerV3 model for state-based observations
    logger.info("Creating DreamerV3 model for state-based input...")
    model = create_world_model(
        "dreamerv3:size12m",
        obs_shape=(obs_dim,),
        action_dim=action_dim,
        # Use MLP encoder instead of CNN for state vectors
        encoder_type="mlp",
        decoder_type="mlp",
        hidden_dim=256,
        deter_dim=256,
        stoch_dim=32,
        stoch_classes=32,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Training config
    config = TrainingConfig(
        total_steps=args.steps,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
        learning_rate=3e-4,
        output_dir=args.output_dir,
        device="auto",
        log_interval=max(1, args.steps // 100),
        save_interval=max(100, args.steps // 10),
    )

    # Callbacks
    callbacks = [
        ProgressCallback(),
        LoggingCallback(),
    ]
    if not args.test:
        callbacks.append(EarlyStoppingCallback(patience=3000))

    # Train
    trainer = Trainer(model, config, callbacks=callbacks)

    logger.info("Starting training...")
    logger.info(f"  Steps: {args.steps}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Sequence length: {args.seq_len}")

    trained_model = trainer.train(buffer)

    # Save final model
    output_path = f"{args.output_dir}/dreamer_mujoco_final"
    logger.info(f"Saving model to {output_path}...")
    trained_model.save_pretrained(output_path)

    logger.info("Training complete!")

    # Run validation
    logger.info("\nRunning validation...")
    validate_dreamer_mujoco(trained_model, buffer, output_dir=args.output_dir)


def validate_dreamer_mujoco(
    model,
    buffer,
    output_dir: str = "./outputs/dreamer_mujoco",
    horizon: int = 20,
):
    """
    Validate DreamerV3 model on MuJoCo state-based data.
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
        initial_obs = obs[:, 0]
        state = model.encode(initial_obs)

        # Run imagination
        action_seq = actions[:, :horizon].permute(1, 0, 2)  # [T, 1, action_dim]
        trajectory = model.rollout(state, action_seq)

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

    # State reconstruction test
    logger.info("\nState reconstruction test:")
    with torch.no_grad():
        for t in [0, 5, 10, min(15, horizon - 1)]:
            real_obs_t = obs[:, t + 1]
            state_t = (
                trajectory.states[t + 1]
                if t + 1 < len(trajectory.states)
                else trajectory.states[-1]
            )
            decoded = model.decode(state_t)
            recon_obs = decoded.preds["obs"]
            recon_mse = torch.mean((real_obs_t - recon_obs) ** 2).item()
            logger.info(f"  t={t}: reconstruction MSE = {recon_mse:.6f}")

    # Latent state analysis
    logger.info("\nLatent state analysis:")

    def _features(state):
        return torch.cat(
            [state.tensors["deter"], state.tensors["stoch"].flatten(1)],
            dim=-1,
        )

    for i, s in enumerate(trajectory.states[:5]):
        features = _features(s)
        norm = features.norm().item()
        logger.info(f"  State {i} feature norm: {norm:.4f}")

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
        corr_str = f"{correlation:.3f}" if "correlation" in dir() else "N/A"
        axes[0, 1].set_title(f"Predicted vs Real (Corr: {corr_str})")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Latent feature norms
        norms = [_features(s).norm().item() for s in trajectory.states[:horizon]]
        axes[1, 0].plot(range(len(norms)), norms, "g-o", linewidth=2, markersize=4)
        axes[1, 0].set_xlabel("Time Step")
        axes[1, 0].set_ylabel("Feature Norm")
        axes[1, 0].set_title("Latent State Evolution")
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: State component comparison
        with torch.no_grad():
            real_obs_0 = obs[:, 0].cpu().numpy().flatten()
            decoded_0 = model.decode(trajectory.states[0]).preds["obs"].cpu().numpy().flatten()

            x = np.arange(len(real_obs_0))
            width = 0.35
            axes[1, 1].bar(x - width / 2, real_obs_0, width, label="Real", alpha=0.7)
            axes[1, 1].bar(x + width / 2, decoded_0, width, label="Reconstructed", alpha=0.7)
            axes[1, 1].set_xlabel("State Dimension")
            axes[1, 1].set_ylabel("Value")
            axes[1, 1].set_title("State Reconstruction (t=0)")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle("DreamerV3 on MuJoCo (State-Based) Validation", fontsize=14)
        plt.tight_layout()

        output_path = Path(output_dir) / "dreamer_mujoco_validation.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved visualization to: {output_path}")
        plt.close()

    except ImportError:
        logger.warning("matplotlib not installed, skipping visualization")


if __name__ == "__main__":
    main()
