# Training API

Complete training infrastructure for world models.

## Quick Start

```python
from worldflux import create_world_model
from worldflux.training import train, ReplayBuffer

model = create_world_model("dreamerv3:size12m", obs_shape=(4,), action_dim=2)
buffer = ReplayBuffer.load("data.npz")

trained_model = train(model, buffer, total_steps=50_000)
```

---

## train

One-liner training function.

```python
from worldflux.training import train

trained_model = train(
    model,
    buffer,
    total_steps=50_000,
    batch_size=16,
    learning_rate=3e-4,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `WorldModel` | required | Model to train |
| `buffer` | `ReplayBuffer` or `BatchProvider` | required | Training data |
| `total_steps` | `int` | `100_000` | Training steps |
| `batch_size` | `int` | `16` | Batch size |
| `sequence_length` | `int` | `50` | Sequence length |
| `learning_rate` | `float` | `3e-4` | Learning rate |

### Returns

Trained model (`nn.Module`)

---

## TrainingConfig

Configuration for training.

```python
from worldflux.training import TrainingConfig

config = TrainingConfig(
    # Duration
    total_steps=100_000,

    # Batch settings
    batch_size=16,
    sequence_length=50,

    # Optimizer
    learning_rate=3e-4,
    weight_decay=0.0,
    grad_clip=100.0,

    # Logging
    log_interval=1000,
)
```

### All Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_steps` | `int` | `100_000` | Total training steps |
| `batch_size` | `int` | `16` | Batch size |
| `sequence_length` | `int` | `50` | Sequence length for BPTT |
| `learning_rate` | `float` | `3e-4` | Adam learning rate |
| `weight_decay` | `float` | `0.0` | L2 regularization |
| `grad_clip` | `float` | `100.0` | Gradient clipping norm |
| `warmup_steps` | `int` | `0` | Learning rate warmup |
| `log_interval` | `int` | `100` | Steps between logging |
| `eval_interval` | `int` | `1000` | Steps between evaluation |
| `save_interval` | `int` | `10000` | Steps between checkpoints |
| `device` | `str` | `"auto"` | Training device |
| `seed` | `int` | `42` | Random seed |

---

## Trainer

Full training control with callbacks.

```python
from worldflux.training import Trainer, TrainingConfig

config = TrainingConfig(total_steps=100_000)
trainer = Trainer(model, config)

trained_model = trainer.train(buffer)
```

### Methods

#### train

```python
trained_model = trainer.train(buffer)
```

#### add_callback

```python
from worldflux.training.callbacks import CheckpointCallback

trainer.add_callback(CheckpointCallback(output_dir="./checkpoints"))
```

---

## ReplayBuffer

Storage for trajectory data.

### Creation

```python
from worldflux.training import ReplayBuffer

buffer = ReplayBuffer(
    capacity=100_000,
    obs_shape=(4,),
    action_dim=2,
)
```

### Adding Data

```python
buffer.add_episode(
    obs=np.array([...]),      # [T, *obs_shape]
    actions=np.array([...]),  # [T, action_dim]
    rewards=np.array([...]),  # [T]
    dones=np.array([...]),    # [T]
)
```

### Sampling

```python
batch = buffer.sample(
    batch_size=16,
    seq_len=50,
    device="cuda",
)

# batch.obs         - [B, T, *obs_shape]
# batch.actions     - [B, T, action_dim]
# batch.rewards     - [B, T]
# batch.terminations - [B, T]
```

### Save/Load

```python
buffer.save("buffer.npz")
buffer = ReplayBuffer.load("buffer.npz")
```

### Required `.npz` Schema

`ReplayBuffer.load()` expects a serialized buffer with these keys:

| Key | Shape | Notes |
|-----|-------|-------|
| `obs` | `[N, *obs_shape]` | Observation rows |
| `actions` | `[N, action_dim]` | Action rows |
| `rewards` | `[N]` | Reward per transition |
| `dones` | `[N]` | Episode termination flags (`0/1`) |
| `obs_shape` | `[len(obs_shape)]` | Stored observation shape metadata |
| `action_dim` | scalar | Stored action-dimension metadata |

If your pipeline uses `terminations`, store the same array under `dones` when
writing the `.npz` artifact for `ReplayBuffer.load()`.

```python
import numpy as np

# Example: valid ReplayBuffer.load() artifact
np.savez(
    "buffer.npz",
    obs=obs.astype(np.float32),                    # [N, *obs_shape]
    actions=actions.astype(np.float32),            # [N, action_dim]
    rewards=rewards.astype(np.float32),            # [N]
    dones=terminations.astype(np.float32),         # [N]
    obs_shape=np.array(obs.shape[1:], dtype=np.int64),
    action_dim=np.array(actions.shape[-1], dtype=np.int64),
)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_episodes` | `int` | Number of episodes |
| `num_transitions` | `int` | Total transitions |
| `__len__` | `int` | Same as num_transitions |

---

## Callbacks

### ProgressCallback

Progress bar with metrics.

```python
from worldflux.training.callbacks import ProgressCallback

trainer.add_callback(ProgressCallback())
```

### LoggingCallback

TensorBoard/console logging.

```python
from worldflux.training.callbacks import LoggingCallback

trainer.add_callback(LoggingCallback(
    log_interval=100,
    use_wandb=False,
))
```

### CheckpointCallback

Periodic model saving.

```python
from worldflux.training.callbacks import CheckpointCallback

trainer.add_callback(CheckpointCallback(
    output_dir="./checkpoints",
    save_interval=10_000,
    max_checkpoints=3,
))
```

### EarlyStoppingCallback

Stop on plateau.

```python
from worldflux.training.callbacks import EarlyStoppingCallback

trainer.add_callback(EarlyStoppingCallback(
    monitor="loss",
    patience=5000,
    min_delta=1e-4,
))
```

### Custom Callbacks

```python
from worldflux.training.callbacks import Callback

class MyCallback(Callback):
    def on_step_end(self, step, losses):
        if step % 1000 == 0:
            print(f"Step {step}: loss={losses['loss']:.4f}")

trainer.add_callback(MyCallback())
```

---

## Data Utilities

### create_random_buffer

Create buffer with random data for testing.

```python
from worldflux.training.data import create_random_buffer

buffer = create_random_buffer(
    capacity=10_000,
    obs_shape=(4,),
    action_dim=2,
    num_episodes=100,
    episode_length=100,
    seed=42,
)
```

---

## Complete Example

```python
from worldflux import create_world_model
from worldflux.training import Trainer, TrainingConfig, ReplayBuffer
from worldflux.training.callbacks import (
    ProgressCallback,
    LoggingCallback,
    CheckpointCallback,
)

# Create model
model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(4,),
    action_dim=2,
    device="cuda",
)

# Load data
buffer = ReplayBuffer.load("trajectories.npz")

# Configure
config = TrainingConfig(
    total_steps=100_000,
    batch_size=16,
    sequence_length=50,
    learning_rate=3e-4,
    log_interval=1000,
)

# Setup trainer
trainer = Trainer(model, config)
trainer.add_callback(ProgressCallback())
trainer.add_callback(LoggingCallback(log_interval=100))
trainer.add_callback(CheckpointCallback(output_dir="./ckpt", save_interval=10000))

# Train
trained_model = trainer.train(buffer)

# Save final
trained_model.save_pretrained("./final_model")
```
