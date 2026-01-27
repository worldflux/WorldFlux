#!/usr/bin/env python3
"""
Collect trajectories from Atari Breakout using random policy.

This script collects data from the Atari Breakout environment using a random
policy for testing DreamerV3 world model training on real visual observations.

Usage:
    # Basic collection
    python examples/collect_atari.py

    # Collect more episodes
    python examples/collect_atari.py --episodes 100 --output breakout_data.npz

    # Use different environment
    python examples/collect_atari.py --env ALE/Pong-v5

Requirements:
    pip install "gymnasium[atari]" ale-py
    autorom --accept-license
"""

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    import ale_py
    import gymnasium as gym

    # Register ALE environments with gymnasium
    gym.register_envs(ale_py)
except ImportError:
    raise ImportError(
        "gymnasium not installed. Install with: pip install 'gymnasium[atari]' ale-py"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_atari(obs: np.ndarray, target_size: tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Preprocess Atari observation from (210, 160, 3) to (3, H, W) normalized.

    Args:
        obs: Raw Atari observation of shape (210, 160, 3).
        target_size: Target height and width (default 64x64).

    Returns:
        Preprocessed observation of shape (3, H, W) in range [0, 1].
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python not installed. Install with: pip install opencv-python")

    # Resize to target size
    resized = cv2.resize(obs, target_size, interpolation=cv2.INTER_AREA)
    # Convert HWC to CHW and normalize to [0, 1]
    return resized.transpose(2, 0, 1).astype(np.float32) / 255.0


def collect_atari_data(
    env_name: str = "ALE/Breakout-v5",
    num_episodes: int = 50,
    output_path: str = "atari_data.npz",
    preprocess: bool = True,
    target_size: tuple[int, int] = (64, 64),
    max_steps_per_episode: int = 10000,
) -> dict[str, np.ndarray]:
    """
    Collect trajectories from Atari with random policy.

    Args:
        env_name: Gymnasium environment name (must be an Atari environment).
        num_episodes: Number of episodes to collect.
        output_path: Path to save the collected data.
        preprocess: Whether to preprocess observations to (3, 64, 64).
        target_size: Target size for preprocessing.
        max_steps_per_episode: Maximum steps per episode to prevent infinite loops.

    Returns:
        Dictionary with collected data.
    """
    logger.info(f"Creating environment: {env_name}")
    env = gym.make(env_name, render_mode=None)

    # Get action space info
    num_actions = env.action_space.n  # type: ignore
    logger.info(f"Action space: {num_actions} discrete actions")

    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    all_rewards: list[float] = []
    all_dones: list[float] = []

    total_steps = 0
    total_reward = 0.0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_obs: list[np.ndarray] = []
        episode_actions: list[int] = []
        episode_rewards: list[float] = []
        episode_dones: list[float] = []
        episode_reward = 0.0

        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            # Random policy
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            if preprocess:
                processed_obs = preprocess_atari(obs, target_size)
                episode_obs.append(processed_obs)
            else:
                episode_obs.append(obs)

            episode_actions.append(action)
            episode_rewards.append(float(reward))
            episode_dones.append(float(done))

            episode_reward += reward
            obs = next_obs
            steps += 1

        # Extend global lists
        all_obs.extend(episode_obs)
        all_actions.extend(episode_actions)
        all_rewards.extend(episode_rewards)
        all_dones.extend(episode_dones)

        total_steps += steps
        total_reward += episode_reward

        logger.info(f"Episode {ep + 1}/{num_episodes}: {steps} steps, reward: {episode_reward:.1f}")

    env.close()

    # Convert to arrays
    if preprocess:
        obs_array = np.array(all_obs, dtype=np.float32)  # Already preprocessed
    else:
        obs_array = np.array(all_obs, dtype=np.uint8)  # Raw observations

    actions_array = np.array(all_actions, dtype=np.int64)
    rewards_array = np.array(all_rewards, dtype=np.float32)
    dones_array = np.array(all_dones, dtype=np.float32)

    # One-hot encode actions
    actions_onehot = np.eye(num_actions, dtype=np.float32)[actions_array]

    logger.info("\nCollection complete:")
    logger.info(f"  Total transitions: {total_steps}")
    logger.info(f"  Total reward: {total_reward:.1f}")
    logger.info(f"  Avg episode length: {total_steps / num_episodes:.1f}")
    logger.info(f"  Avg episode reward: {total_reward / num_episodes:.1f}")
    logger.info(f"  Obs shape: {obs_array.shape}")

    # Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        obs=obs_array,
        actions=actions_array,
        actions_onehot=actions_onehot,
        rewards=rewards_array,
        dones=dones_array,
        num_actions=np.array(num_actions),
        env_name=np.array(env_name),
    )
    logger.info(f"Saved to: {output_path}")

    return {
        "obs": obs_array,
        "actions": actions_array,
        "actions_onehot": actions_onehot,
        "rewards": rewards_array,
        "dones": dones_array,
    }


def load_atari_to_buffer(
    path: str,
    capacity: int | None = None,
):
    """
    Load collected Atari data into a ReplayBuffer.

    Args:
        path: Path to the .npz file created by collect_atari_data.
        capacity: Buffer capacity. If None, uses data size.

    Returns:
        ReplayBuffer ready for training.
    """
    from worldloom.training import ReplayBuffer

    logger.info(f"Loading data from: {path}")
    data = np.load(path, allow_pickle=False)

    obs = data["obs"]
    actions_onehot = data["actions_onehot"]
    rewards = data["rewards"]
    dones = data["dones"]

    logger.info(f"Loaded {len(obs)} transitions")
    logger.info(f"Obs shape: {obs.shape}, Actions shape: {actions_onehot.shape}")

    # Determine capacity
    if capacity is None:
        capacity = len(obs)

    # Create buffer
    obs_shape = obs.shape[1:]  # (3, 64, 64) for preprocessed
    action_dim = actions_onehot.shape[1]  # num_actions

    buffer = ReplayBuffer(
        capacity=capacity,
        obs_shape=obs_shape,
        action_dim=action_dim,
    )

    # Split into episodes based on done flags
    episode_starts = [0] + list(np.where(dones[:-1] == 1.0)[0] + 1)
    episode_ends = list(np.where(dones == 1.0)[0] + 1)

    if len(episode_ends) < len(episode_starts):
        episode_ends.append(len(obs))

    num_episodes = min(len(episode_starts), len(episode_ends))
    logger.info(f"Found {num_episodes} episodes")

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


def main():
    parser = argparse.ArgumentParser(description="Collect Atari trajectories")
    parser.add_argument(
        "--env",
        type=str,
        default="ALE/Breakout-v5",
        help="Atari environment name",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="atari_data.npz",
        help="Output path for collected data",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Save raw observations without preprocessing",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Target observation size (H=W)",
    )
    args = parser.parse_args()

    collect_atari_data(
        env_name=args.env,
        num_episodes=args.episodes,
        output_path=args.output,
        preprocess=not args.no_preprocess,
        target_size=(args.size, args.size),
    )


if __name__ == "__main__":
    main()
