#!/usr/bin/env python3
"""
Create demo GIFs for WorldFlux documentation.

This script generates visualization GIFs showing:
1. Imagination rollout comparison (real vs imagined)
2. Latent space visualization
3. Architecture animation

Usage:
    # Basic usage (generates all demos with random data)
    python scripts/create_demo_gif.py

    # With a trained model
    python scripts/create_demo_gif.py --model ./outputs/my_model --data ./data.npz

    # Specific output directory
    python scripts/create_demo_gif.py --output ./docs/assets

Requirements:
    pip install matplotlib imageio pillow
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


def create_imagination_gif(
    model,
    buffer,
    output_path: str = "imagination_rollout.gif",
    horizon: int = 30,
    fps: int = 5,
):
    """
    Create a GIF comparing real vs imagined observations.

    Shows side-by-side comparison of actual environment frames
    and frames imagined by the world model.
    """
    try:
        import imageio
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for rendering
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install: pip install matplotlib imageio pillow")

    device = next(model.parameters()).device
    logger.info(f"Creating imagination GIF with horizon={horizon}")

    # Sample a sequence
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

    # Get real and predicted observations
    real_obs = obs[:, 1 : horizon + 1].cpu()  # [1, T, C, H, W]
    pred_obs = torch.stack(imagined_obs[1:], dim=1)  # [1, T, C, H, W]

    # Create frames
    frames = []
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    for t in range(horizon):
        # Clear previous
        for ax in axes:
            ax.clear()

        # Real observation
        if t < real_obs.shape[1]:
            real_img = real_obs[0, t].permute(1, 2, 0).numpy()
            real_img = np.clip(real_img, 0, 1)
            axes[0].imshow(real_img)
        axes[0].set_title(f"Real (t={t})", fontsize=14)
        axes[0].axis("off")

        # Imagined observation
        if t < pred_obs.shape[1]:
            pred_img = pred_obs[0, t].permute(1, 2, 0).numpy()
            pred_img = np.clip(pred_img, 0, 1)
            axes[1].imshow(pred_img)
        axes[1].set_title(f"Imagined (t={t})", fontsize=14)
        axes[1].axis("off")

        # Add reward annotation
        if t < len(trajectory.rewards):
            reward = trajectory.rewards[t, 0].item()
            fig.suptitle(f"WorldFlux Imagination | Predicted Reward: {reward:.2f}", fontsize=12)

        plt.tight_layout()

        # Convert to image
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(img)

    plt.close()

    # Save GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    logger.info(f"Saved imagination GIF to: {output_path}")

    return output_path


def create_latent_space_gif(
    model,
    buffer,
    output_path: str = "latent_space.gif",
    num_trajectories: int = 5,
    horizon: int = 50,
    fps: int = 10,
):
    """
    Create a GIF showing latent space trajectories.

    Visualizes how different episodes traverse the latent space
    using PCA for 2D projection.
    """
    try:
        import imageio
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for rendering
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        raise ImportError("Please install: pip install matplotlib imageio scikit-learn pillow")

    device = next(model.parameters()).device
    logger.info(f"Creating latent space GIF with {num_trajectories} trajectories")

    # Collect latent trajectories
    all_trajectories = []

    for _ in range(num_trajectories):
        batch = buffer.sample(batch_size=1, seq_len=horizon, device=device)
        obs = batch["obs"]

        trajectory = []
        with torch.no_grad():
            for t in range(horizon):
                state = model.encode(obs[:, t])
                trajectory.append(state.features.cpu().numpy().flatten())

        all_trajectories.append(np.array(trajectory))

    # Stack all points for PCA
    all_points = np.concatenate(all_trajectories, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_points)

    # Transform trajectories
    transformed = [pca.transform(traj) for traj in all_trajectories]

    # Color map for trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, num_trajectories))

    # Create frames
    frames = []
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calculate axis limits
    all_transformed = np.concatenate(transformed, axis=0)
    x_min, x_max = all_transformed[:, 0].min(), all_transformed[:, 0].max()
    y_min, y_max = all_transformed[:, 1].min(), all_transformed[:, 1].max()
    margin = 0.1 * max(x_max - x_min, y_max - y_min)

    for t in range(horizon):
        ax.clear()

        for i, (traj, color) in enumerate(zip(transformed, colors)):
            # Plot trajectory up to current time
            if t > 0:
                ax.plot(
                    traj[: t + 1, 0],
                    traj[: t + 1, 1],
                    c=color,
                    alpha=0.5,
                    linewidth=2,
                )

            # Plot current point
            ax.scatter(
                traj[t, 0],
                traj[t, 1],
                c=[color],
                s=100,
                marker="o",
                edgecolors="black",
                linewidth=1,
            )

            # Plot start point
            ax.scatter(
                traj[0, 0],
                traj[0, 1],
                c=[color],
                s=200,
                marker="*",
                edgecolors="black",
                linewidth=1,
            )

        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel("PC1", fontsize=12)
        ax.set_ylabel("PC2", fontsize=12)
        ax.set_title(f"Latent Space Trajectories (t={t})", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Add variance explained
        var_explained = sum(pca.explained_variance_ratio_) * 100
        ax.text(
            0.02,
            0.98,
            f"Variance explained: {var_explained:.1f}%",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
        )

        plt.tight_layout()

        # Convert to image
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(img)

    plt.close()

    # Save GIF
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    logger.info(f"Saved latent space GIF to: {output_path}")

    return output_path


def create_architecture_diagram(output_path: str = "architecture.png"):
    """
    Create a static architecture diagram.

    Shows the data flow through the world model components.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for rendering
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install: pip install matplotlib")

    logger.info("Creating architecture diagram")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Colors
    colors = {
        "input": "#E3F2FD",
        "encoder": "#BBDEFB",
        "latent": "#90CAF9",
        "dynamics": "#FFE0B2",
        "decoder": "#C8E6C9",
        "output": "#F5F5F5",
    }

    def add_box(x, y, w, h, text, color, fontsize=10):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
        )

    def add_arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="black", lw=2),
        )

    # Boxes
    add_box(0.5, 2.5, 1.5, 1, "Observation", colors["input"])
    add_box(2.5, 2.5, 1.5, 1, "Encoder", colors["encoder"])
    add_box(4.5, 2.5, 2, 1, "LatentState", colors["latent"])
    add_box(7, 2.5, 1.5, 1, "Dynamics", colors["dynamics"])
    add_box(9, 2.5, 1.5, 1, "Decoder", colors["decoder"])
    add_box(9, 0.5, 1.5, 1.5, "Predictions\nobs, reward\ncontinue", colors["output"], fontsize=9)

    add_box(4.5, 4.5, 1, 0.8, "Action", colors["input"])

    # Arrows
    add_arrow(2, 3, 2.5, 3)
    add_arrow(4, 3, 4.5, 3)
    add_arrow(6.5, 3, 7, 3)
    add_arrow(8.5, 3, 9, 3)
    add_arrow(9.75, 2.5, 9.75, 2)

    # Dynamics loop
    add_arrow(8.5, 3, 8.5, 4)
    add_arrow(8.5, 4, 6.5, 4)
    add_arrow(6.5, 4, 6.5, 3.5)
    add_arrow(5.5, 4.5, 6.5, 4)

    # Title
    ax.text(
        6,
        5.5,
        "WorldFlux Architecture",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved architecture diagram to: {output_path}")
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create demo GIFs for WorldFlux")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to trained model (uses random model if not specified)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (uses random data if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./docs/assets",
        help="Output directory for GIFs",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Imagination horizon",
    )
    parser.add_argument(
        "--skip-gifs",
        action="store_true",
        help="Skip GIF generation (only create static images)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always create architecture diagram
    create_architecture_diagram(output_dir / "architecture.png")

    if args.skip_gifs:
        logger.info("Skipping GIF generation")
        return

    # Create or load model
    from worldflux import create_world_model
    from worldflux.training import ReplayBuffer
    from worldflux.training.data import create_random_buffer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model:
        logger.info(f"Loading model from: {args.model}")
        model = create_world_model(args.model, device=device)
        obs_shape = model.config.obs_shape
        action_dim = model.config.action_dim
    else:
        logger.info("Creating random DreamerV3 model for demo")
        obs_shape = (3, 64, 64)
        action_dim = 4
        model = create_world_model(
            "dreamerv3:size12m",
            obs_shape=obs_shape,
            action_dim=action_dim,
            device=device,
        )

    model.eval()

    # Create or load buffer
    if args.data:
        logger.info(f"Loading data from: {args.data}")
        data = np.load(args.data)
        buffer = ReplayBuffer(
            capacity=len(data["obs"]),
            obs_shape=obs_shape,
            action_dim=action_dim,
        )
        # Use one-hot actions if available (2D), otherwise use raw actions
        if "actions_onehot" in data:
            actions = data["actions_onehot"][:100]
        else:
            actions = data["actions"][:100]
            # Ensure actions are 2D [T, action_dim]
            if actions.ndim == 1:
                actions = np.eye(action_dim, dtype=np.float32)[actions]

        # Add data as episodes (simplified - assumes contiguous)
        buffer.add_episode(
            obs=data["obs"][:100],
            actions=actions,
            rewards=data["rewards"][:100],
            dones=data["dones"][:100],
        )
    else:
        logger.info("Creating random buffer for demo")
        buffer = create_random_buffer(
            capacity=5000,
            obs_shape=obs_shape,
            action_dim=action_dim,
            num_episodes=50,
            episode_length=100,
            seed=42,
        )

    # Create GIFs
    try:
        create_imagination_gif(
            model,
            buffer,
            output_path=output_dir / "imagination_rollout.gif",
            horizon=args.horizon,
        )
    except Exception as e:
        logger.warning(f"Failed to create imagination GIF: {e}")

    try:
        create_latent_space_gif(
            model,
            buffer,
            output_path=output_dir / "latent_space.gif",
            horizon=50,
        )
    except Exception as e:
        logger.warning(f"Failed to create latent space GIF: {e}")

    logger.info(f"\nAll demos saved to: {output_dir}")


if __name__ == "__main__":
    main()
