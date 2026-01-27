#!/usr/bin/env python3
"""
Collect trajectories from MuJoCo continuous control environments.

This script collects data from MuJoCo environments using a random policy
for testing TD-MPC2 world model training on continuous control tasks.

Usage:
    # Basic collection (HalfCheetah)
    python examples/collect_mujoco.py

    # Different environment
    python examples/collect_mujoco.py --env Hopper-v5 --episodes 100

    # List available environments
    python examples/collect_mujoco.py --list-envs

Requirements:
    pip install "gymnasium[mujoco]"
"""

import argparse
import logging
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    raise ImportError("gymnasium not installed. Install with: pip install 'gymnasium[mujoco]'")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Popular MuJoCo environments for benchmarking
MUJOCO_ENVS = [
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Ant-v5",
    "Humanoid-v5",
    "Swimmer-v5",
    "Reacher-v5",
    "Pusher-v5",
    "InvertedPendulum-v5",
    "InvertedDoublePendulum-v5",
]


def list_mujoco_envs():
    """List available MuJoCo environments."""
    print("\nAvailable MuJoCo environments:")
    print("-" * 40)
    for env_name in MUJOCO_ENVS:
        try:
            env = gym.make(env_name)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            env.close()
            print(f"  {env_name:<30} obs={obs_dim}, act={act_dim}")
        except Exception as e:
            print(f"  {env_name:<30} (not available: {e})")
    print()


def collect_mujoco_data(
    env_name: str = "HalfCheetah-v5",
    num_episodes: int = 100,
    output_path: str = "mujoco_data.npz",
    max_steps_per_episode: int = 1000,
    action_noise: float = 0.1,
) -> dict[str, np.ndarray]:
    """
    Collect trajectories from MuJoCo with random/noisy policy.

    Args:
        env_name: Gymnasium MuJoCo environment name.
        num_episodes: Number of episodes to collect.
        output_path: Path to save the collected data.
        max_steps_per_episode: Maximum steps per episode.
        action_noise: Standard deviation of action noise (0 = pure random).

    Returns:
        Dictionary with collected data.
    """
    logger.info(f"Creating environment: {env_name}")

    try:
        env = gym.make(env_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to create environment '{env_name}'. "
            f"Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'\n"
            f"Error: {e}"
        )

    # Get space info
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    logger.info(f"Observation dim: {obs_dim}")
    logger.info(f"Action dim: {act_dim}")
    logger.info(f"Action range: [{act_low[0]:.2f}, {act_high[0]:.2f}]")

    all_obs: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_rewards: list[float] = []
    all_dones: list[float] = []

    total_steps = 0
    total_reward = 0.0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_obs: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []
        episode_rewards: list[float] = []
        episode_dones: list[float] = []
        episode_reward = 0.0

        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            # Random action with optional noise
            if action_noise > 0:
                # Sample around zero with noise
                action = np.random.randn(act_dim) * action_noise
            else:
                # Uniform random in action space
                action = np.random.uniform(act_low, act_high)

            # Clip to valid range
            action = np.clip(action, act_low, act_high)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store transition
            episode_obs.append(obs.astype(np.float32))
            episode_actions.append(action.astype(np.float32))
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

        if (ep + 1) % 10 == 0 or ep == 0:
            logger.info(
                f"Episode {ep + 1}/{num_episodes}: {steps} steps, reward: {episode_reward:.1f}"
            )

    env.close()

    # Convert to arrays
    obs_array = np.array(all_obs, dtype=np.float32)
    actions_array = np.array(all_actions, dtype=np.float32)
    rewards_array = np.array(all_rewards, dtype=np.float32)
    dones_array = np.array(all_dones, dtype=np.float32)

    logger.info("\nCollection complete:")
    logger.info(f"  Total transitions: {total_steps}")
    logger.info(f"  Total reward: {total_reward:.1f}")
    logger.info(f"  Avg episode length: {total_steps / num_episodes:.1f}")
    logger.info(f"  Avg episode reward: {total_reward / num_episodes:.1f}")
    logger.info(f"  Obs shape: {obs_array.shape}")
    logger.info(f"  Actions shape: {actions_array.shape}")

    # Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        obs=obs_array,
        actions=actions_array,
        rewards=rewards_array,
        dones=dones_array,
        obs_dim=np.array(obs_dim),
        action_dim=np.array(act_dim),
        env_name=np.array(env_name),
    )
    logger.info(f"Saved to: {output_path}")

    return {
        "obs": obs_array,
        "actions": actions_array,
        "rewards": rewards_array,
        "dones": dones_array,
    }


def load_mujoco_to_buffer(
    path: str,
    capacity: int | None = None,
):
    """
    Load collected MuJoCo data into a ReplayBuffer.

    Args:
        path: Path to the .npz file created by collect_mujoco_data.
        capacity: Buffer capacity. If None, uses data size.

    Returns:
        ReplayBuffer ready for training.
    """
    from worldloom.training import ReplayBuffer

    logger.info(f"Loading data from: {path}")
    data = np.load(path, allow_pickle=False)

    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    obs_dim = int(data["obs_dim"])
    action_dim = int(data["action_dim"])

    logger.info(f"Loaded {len(obs)} transitions")
    logger.info(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

    # Determine capacity
    if capacity is None:
        capacity = len(obs)

    # Create buffer
    buffer = ReplayBuffer(
        capacity=capacity,
        obs_shape=(obs_dim,),
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
            actions=actions[start:end],
            rewards=rewards[start:end],
            dones=dones[start:end],
        )

    logger.info(f"Buffer: {len(buffer)} transitions, {buffer.num_episodes} episodes")
    return buffer


def main():
    parser = argparse.ArgumentParser(description="Collect MuJoCo trajectories")
    parser.add_argument(
        "--env",
        type=str,
        default="HalfCheetah-v5",
        help="MuJoCo environment name",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mujoco_data.npz",
        help="Output path for collected data",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--action-noise",
        type=float,
        default=0.3,
        help="Action noise std (0 for uniform random)",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List available MuJoCo environments",
    )
    args = parser.parse_args()

    if args.list_envs:
        list_mujoco_envs()
        return

    collect_mujoco_data(
        env_name=args.env,
        num_episodes=args.episodes,
        output_path=args.output,
        max_steps_per_episode=args.max_steps,
        action_noise=args.action_noise,
    )


if __name__ == "__main__":
    main()
