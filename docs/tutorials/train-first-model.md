# Train Your First Model

A complete walkthrough of training a world model from scratch.

## Prerequisites

```bash
pip install -e ".[training]"
```

## Step 1: Prepare Your Data

World models learn from trajectories: sequences of observations, actions, and rewards.

### Option A: Collect from Gym Environment

```python
import gymnasium as gym
import numpy as np
from worldflux.training import ReplayBuffer

# Create buffer
buffer = ReplayBuffer(
    capacity=100_000,
    obs_shape=(4,),
    action_dim=2,
)

# Collect trajectories
env = gym.make("CartPole-v1")

for episode in range(100):
    obs_list, action_list, reward_list, done_list = [], [], [], []
    obs, _ = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_list.append(obs)
        action_list.append([action])  # Make it 2D
        reward_list.append(reward)
        done_list.append(done)

        obs = next_obs

    buffer.add_episode(
        obs=np.array(obs_list),
        actions=np.array(action_list),
        rewards=np.array(reward_list),
        dones=np.array(done_list),
    )

# Save for later
buffer.save("cartpole_data.npz")
```

### Option B: Use Random Data (Testing)

```python
from worldflux.training.data import create_random_buffer

buffer = create_random_buffer(
    capacity=10_000,
    obs_shape=(4,),
    action_dim=2,
    num_episodes=100,
    episode_length=100,
)
```

## Step 2: Create a Model

```python
from worldflux import create_world_model

model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(4,),
    action_dim=2,
    device="cuda",  # or "cpu"
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Step 3: Configure Training

```python
from worldflux.training import TrainingConfig

config = TrainingConfig(
    # Duration
    total_steps=50_000,

    # Batch settings
    batch_size=16,
    sequence_length=50,

    # Optimizer
    learning_rate=3e-4,
    grad_clip=100.0,

    # Logging
    log_interval=1000,
)
```

## Step 4: Train

### Simple (One-Liner)

```python
from worldflux.training import train

trained_model = train(model, buffer, total_steps=50_000)
```

### With Full Control

```python
from worldflux.training import Trainer
from worldflux.training.callbacks import (
    LoggingCallback,
    CheckpointCallback,
    ProgressCallback,
)

trainer = Trainer(model, config)

# Add callbacks
trainer.add_callback(ProgressCallback())
trainer.add_callback(LoggingCallback(log_dir="./logs"))
trainer.add_callback(CheckpointCallback(
    save_dir="./checkpoints",
    save_every=10_000,
))

# Train
trained_model = trainer.train(buffer)
```

## Step 5: Evaluate

```python
import torch

model.eval()

# Sample a test batch
batch = buffer.sample(batch_size=4, seq_len=20, device="cuda")

# Compute losses
with torch.no_grad():
    loss_out = model.loss(batch)

print("Final losses:")
print(f"  total: {loss_out.loss.item():.4f}")
for name, value in loss_out.components.items():
    print(f"  {name}: {value.item():.4f}")
```

## Step 6: Use the Model

### Imagination Rollout

```python
# Encode initial observation
obs = batch.obs[:, 0]  # First observation
state = model.encode(obs)

# Generate action sequence
actions = torch.randn(15, 4, 2, device="cuda")

# Imagine future
trajectory = model.rollout(state, actions)

print(f"Predicted rewards: {trajectory.rewards.mean():.4f}")
```

### Save and Load

```python
# Save
model.save_pretrained("./my_trained_model")

# Load
loaded_model = create_world_model("./my_trained_model")
```

## Reproduce with Bundled Data

If you want a **shortest real-data run** with bundled datasets, see:

- [Reproduce Dreamer/TD-MPC2](reproduce-dreamer-tdmpc2.md)

## Tips

### Memory Issues

- Use smaller model: `dreamerv3:size12m` instead of `size200m`
- Reduce batch size: `batch_size=8`
- Reduce sequence length: `sequence_length=25`

### Slow Training

- Enable mixed precision (automatic with PyTorch 2.0+)
- Use GPU: `device="cuda"`
- Increase batch size if memory allows

### Poor Results

- Collect more data (100+ episodes)
- Train longer (50K+ steps)
- Check your data quality (rewards, episode boundaries)

## Next Steps

- [DreamerV3 vs TD-MPC2](dreamer-vs-tdmpc2.md) - Choose the right model
- [API Reference](../api/training.md) - Training configuration options
