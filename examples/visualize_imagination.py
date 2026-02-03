#!/usr/bin/env python3
"""
Visualize imagination rollouts from a trained DreamerV3 model.

This script loads a trained model and generates visualizations comparing
real observations with imagined observations over a multi-step horizon.

Usage:
    # Basic visualization
    python examples/visualize_imagination.py --model ./outputs/atari_dreamer/atari_dreamer_final

    # With custom data
    python examples/visualize_imagination.py --model ./outputs/atari_dreamer/atari_dreamer_final --data atari_data.npz

    # Longer horizon
    python examples/visualize_imagination.py --model ./outputs/atari_dreamer/atari_dreamer_final --horizon 50

Requirements:
    pip install matplotlib
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load a trained DreamerV3 model."""
    import os

    from worldflux import create_world_model

    logger.info(f"Loading model from: {model_path}")

    # Check if path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please train a model first with:\n"
            f"  python examples/train_atari_dreamer.py --data atari_data.npz"
        )

    # Check for required files
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"The model directory exists but is missing config.json."
        )

    model = create_world_model(model_path)
    model.eval()
    return model


def load_buffer(data_path: str | None, obs_shape: tuple, action_dim: int):
    """Load data into a ReplayBuffer or create random data for testing."""
    from worldflux.training import ReplayBuffer
    from worldflux.training.data import create_random_buffer

    # Use random data if no path provided or if path doesn't exist
    if data_path is None or data_path.lower() == "random" or not Path(data_path).exists():
        if data_path is not None and data_path.lower() != "random" and not Path(data_path).exists():
            logger.warning(f"Data file not found: {data_path}, using random data instead")
        logger.info("Creating random data for visualization...")
        return create_random_buffer(
            capacity=5000,
            obs_shape=obs_shape,
            action_dim=action_dim,
            num_episodes=50,
            episode_length=100,
            seed=42,
        )

    logger.info(f"Loading data from: {data_path}")
    data = np.load(data_path, allow_pickle=False)

    obs = data["obs"]
    actions_onehot = data["actions_onehot"]
    rewards = data["rewards"]
    dones = data["dones"]

    capacity = len(obs)
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


def visualize_single_rollout(
    model,
    buffer,
    horizon: int = 20,
    output_path: str = "imagination_rollout.png",
):
    """Generate visualization of a single imagination rollout."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib not installed. Install with: pip install matplotlib")

    device = next(model.parameters()).device

    # Sample a batch
    batch = buffer.sample(batch_size=1, seq_len=horizon + 1, device=device)
    obs = batch["obs"]  # [1, T+1, C, H, W]
    actions = batch["actions"]  # [1, T+1, A]

    with torch.no_grad():
        # Encode initial observation
        initial_obs = obs[:, 0]
        state = model.encode(initial_obs)

        # Run imagination
        action_seq = actions[:, :horizon].permute(1, 0, 2)  # [T, 1, A]
        trajectory = model.imagine(state, action_seq)

        # Decode imagined observations
        imagined_obs = []
        for s in trajectory.states:
            decoded = model.decode(s)
            imagined_obs.append(decoded["obs"].cpu())

    # Get real observations
    real_obs = obs[:, 1 : horizon + 1].cpu()  # [1, T, C, H, W]
    pred_obs = torch.stack(imagined_obs[1:], dim=1)  # [1, T, C, H, W]

    # Compute metrics
    mse = torch.mean((real_obs - pred_obs) ** 2).item()

    # Create visualization
    num_frames = 6
    fig, axes = plt.subplots(2, num_frames, figsize=(3 * num_frames, 6))

    frame_indices = [int(i * (horizon - 1) / (num_frames - 1)) for i in range(num_frames)]

    # Top row: Real observations
    for col, t in enumerate(frame_indices):
        if t < real_obs.shape[1]:
            img = real_obs[0, t].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[0, col].imshow(img)
        axes[0, col].set_title(f"t={t}")
        axes[0, col].axis("off")
        if col == 0:
            axes[0, col].set_ylabel("Real", fontsize=12)

    # Bottom row: Imagined observations
    for col, t in enumerate(frame_indices):
        if t < pred_obs.shape[1]:
            img = pred_obs[0, t].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            axes[1, col].imshow(img)
        axes[1, col].axis("off")
        if col == 0:
            axes[1, col].set_ylabel("Imagined", fontsize=12)

    plt.suptitle(f"Imagination Rollout (Horizon={horizon}, MSE={mse:.4f})", fontsize=14)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved visualization to: {output_path}")
    plt.close()

    return mse


def visualize_reward_prediction(
    model,
    buffer,
    horizon: int = 50,
    num_rollouts: int = 10,
    output_path: str = "reward_prediction.png",
):
    """Visualize reward prediction accuracy over multiple rollouts."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib not installed. Install with: pip install matplotlib")

    device = next(model.parameters()).device

    all_real_rewards = []
    all_pred_rewards = []

    for _ in range(num_rollouts):
        batch = buffer.sample(batch_size=1, seq_len=horizon + 1, device=device)
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]

        with torch.no_grad():
            initial_obs = obs[:, 0]
            state = model.encode(initial_obs)
            action_seq = actions[:, :horizon].permute(1, 0, 2)
            trajectory = model.imagine(state, action_seq)

        real_rewards = rewards[:, 1 : horizon + 1].cpu().numpy().flatten()
        pred_rewards = trajectory.rewards.cpu().numpy().flatten()

        all_real_rewards.append(real_rewards)
        all_pred_rewards.append(pred_rewards)

    # Stack results
    all_real = np.array(all_real_rewards)  # [num_rollouts, horizon]
    all_pred = np.array(all_pred_rewards)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean reward over time
    mean_real = np.mean(all_real, axis=0)
    std_real = np.std(all_real, axis=0)
    mean_pred = np.mean(all_pred, axis=0)
    std_pred = np.std(all_pred, axis=0)

    t = np.arange(horizon)
    axes[0].plot(t, mean_real, label="Real", color="blue", linewidth=2)
    axes[0].fill_between(t, mean_real - std_real, mean_real + std_real, alpha=0.2, color="blue")
    axes[0].plot(t, mean_pred, label="Predicted", color="red", linewidth=2)
    axes[0].fill_between(t, mean_pred - std_pred, mean_pred + std_pred, alpha=0.2, color="red")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward Over Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Scatter plot of predicted vs real
    axes[1].scatter(all_real.flatten(), all_pred.flatten(), alpha=0.3, s=10)
    lims = [
        min(all_real.min(), all_pred.min()),
        max(all_real.max(), all_pred.max()),
    ]
    axes[1].plot(lims, lims, "r--", linewidth=2, label="Perfect prediction")
    axes[1].set_xlabel("Real Reward")
    axes[1].set_ylabel("Predicted Reward")
    axes[1].set_title("Predicted vs Real Reward")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Compute correlation
    correlation = np.corrcoef(all_real.flatten(), all_pred.flatten())[0, 1]
    mse = np.mean((all_real - all_pred) ** 2)
    plt.suptitle(f"Reward Prediction (Correlation: {correlation:.3f}, MSE: {mse:.4f})", fontsize=14)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved reward visualization to: {output_path}")
    plt.close()

    return correlation, mse


def visualize_latent_dynamics(
    model,
    buffer,
    horizon: int = 100,
    output_path: str = "latent_dynamics.png",
):
    """Visualize latent state dynamics using PCA."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError(
            "matplotlib and scikit-learn required. "
            "Install with: pip install matplotlib scikit-learn"
        )

    device = next(model.parameters()).device

    batch = buffer.sample(batch_size=1, seq_len=horizon, device=device)
    obs = batch["obs"]
    actions = batch["actions"]

    # Collect real latent states
    real_latents = []
    with torch.no_grad():
        for t in range(horizon):
            state = model.encode(obs[:, t])
            real_latents.append(state.features.cpu().numpy())

    real_latents = np.concatenate(real_latents, axis=0)  # [horizon, latent_dim]

    # Collect imagined latent states
    with torch.no_grad():
        initial_state = model.encode(obs[:, 0])
        action_seq = actions[:, : horizon - 1].permute(1, 0, 2)
        trajectory = model.imagine(initial_state, action_seq)

    imagined_latents = []
    for s in trajectory.states:
        imagined_latents.append(s.features.cpu().numpy())
    imagined_latents = np.concatenate(imagined_latents, axis=0)

    # PCA
    all_latents = np.concatenate([real_latents, imagined_latents], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_latents)

    real_pca = pca.transform(real_latents)
    imagined_pca = pca.transform(imagined_latents)

    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot real trajectory
    ax.plot(real_pca[:, 0], real_pca[:, 1], "b-", alpha=0.5, linewidth=1, label="Real")
    ax.scatter(real_pca[:, 0], real_pca[:, 1], c=np.arange(len(real_pca)), cmap="Blues", s=20)

    # Plot imagined trajectory
    ax.plot(imagined_pca[:, 0], imagined_pca[:, 1], "r-", alpha=0.5, linewidth=1, label="Imagined")
    ax.scatter(
        imagined_pca[:, 0], imagined_pca[:, 1], c=np.arange(len(imagined_pca)), cmap="Reds", s=20
    )

    # Mark start points
    ax.scatter(real_pca[0, 0], real_pca[0, 1], c="blue", s=200, marker="*", zorder=5)
    ax.scatter(imagined_pca[0, 0], imagined_pca[0, 1], c="red", s=200, marker="*", zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        f"Latent Space Dynamics (PCA, variance explained: {sum(pca.explained_variance_ratio_):.1%})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved latent dynamics visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize DreamerV3 imagination rollouts")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="random",
        help="Path to collected data, or 'random' for generated test data",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Imagination horizon",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/visualizations",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all visualizations (rollout, reward, latent)",
    )
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Get model config
    obs_shape = model.config.obs_shape
    action_dim = model.config.action_dim

    # Load buffer
    buffer = load_buffer(args.data, obs_shape, action_dim)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    logger.info("Generating imagination rollout visualization...")
    mse = visualize_single_rollout(
        model,
        buffer,
        horizon=args.horizon,
        output_path=output_dir / "imagination_rollout.png",
    )
    logger.info(f"Imagination MSE: {mse:.6f}")

    if args.all:
        logger.info("Generating reward prediction visualization...")
        corr, reward_mse = visualize_reward_prediction(
            model,
            buffer,
            horizon=args.horizon,
            output_path=output_dir / "reward_prediction.png",
        )
        logger.info(f"Reward correlation: {corr:.3f}, MSE: {reward_mse:.6f}")

        logger.info("Generating latent dynamics visualization...")
        try:
            visualize_latent_dynamics(
                model,
                buffer,
                horizon=100,
                output_path=output_dir / "latent_dynamics.png",
            )
        except ImportError as e:
            logger.warning(f"Skipping latent dynamics: {e}")

    logger.info(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
