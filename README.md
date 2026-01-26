# World Models SDK

[![CI](https://github.com/worldmodels/worldmodels-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/worldmodels/worldmodels-sdk/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/worldmodels/worldmodels-sdk/branch/main/graph/badge.svg)](https://codecov.io/gh/worldmodels/worldmodels-sdk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

A unified Python interface for latent world models used in reinforcement learning.

## Features

- **Unified API**: Common interface across DreamerV3, TD-MPC2, and more
- **Simple Usage**: One-liner model creation with `create_world_model()`
- **Training Infrastructure**: Complete training loop with callbacks, checkpointing, and logging
- **Type Safe**: Full type annotations and mypy compatibility

## Installation

```bash
pip install worldmodels

# With training dependencies
pip install worldmodels[training]

# With all optional dependencies
pip install worldmodels[all]
```

For development:

```bash
git clone https://github.com/worldmodels/worldmodels-sdk
cd worldmodels-sdk
pip install -e ".[dev]"
```

## Quick Start

### Create a Model

```python
from worldmodels import create_world_model

# DreamerV3 (image observations)
model = create_world_model("dreamerv3:size12m")

# TD-MPC2 (state observations)
model = create_world_model("tdmpc2:5m", obs_shape=(39,), action_dim=6)
```

### Train a Model

```python
from worldmodels import create_world_model
from worldmodels.training import train, ReplayBuffer

# Create model
model = create_world_model("dreamerv3:size12m", obs_shape=(4,), action_dim=2)

# Load data
buffer = ReplayBuffer.load("trajectories.npz")

# Train (one-liner)
trained_model = train(model, buffer, total_steps=50_000)

# Save
trained_model.save_pretrained("./my_model")
```

### Full Training Control

```python
from worldmodels import create_world_model
from worldmodels.training import Trainer, TrainingConfig, ReplayBuffer

model = create_world_model("tdmpc2:5m", obs_shape=(39,), action_dim=6)
buffer = ReplayBuffer(capacity=100_000, obs_shape=(39,), action_dim=6)

config = TrainingConfig(
    total_steps=100_000,
    batch_size=256,
    learning_rate=1e-4,
)

trainer = Trainer(model, config)
trainer.train(buffer)
```

## Available Models

### DreamerV3

| Preset | Parameters | Description |
|--------|------------|-------------|
| `dreamerv3:size12m` | 12M | Small, fast training |
| `dreamerv3:size25m` | 25M | Balanced |
| `dreamerv3:size50m` | 50M | Standard |
| `dreamerv3:size100m` | 100M | Large |
| `dreamerv3:size200m` | 200M | Maximum capacity |

### TD-MPC2

| Preset | Parameters | Description |
|--------|------------|-------------|
| `tdmpc2:5m` | 5M | Small, fast |
| `tdmpc2:19m` | 19M | Balanced |
| `tdmpc2:48m` | 48M | Large |
| `tdmpc2:317m` | 317M | Maximum capacity |

## API Reference

### Core Methods

All world models implement the `WorldModel` protocol:

```python
# Encode observation to latent state
state = model.encode(obs)

# Predict next state (imagination, no observation)
next_state = model.predict(state, action)

# Update state with observation (posterior)
next_state = model.observe(state, action, obs)

# Decode latent state to predictions
predictions = model.decode(state)  # {"obs", "reward", "continue"}

# Multi-step imagination rollout
trajectory = model.imagine(initial_state, actions)

# Compute training losses
losses = model.compute_loss(batch)  # {"loss", "kl", "reconstruction", ...}
```

### Training API

```python
from worldmodels.training import (
    Trainer,
    TrainingConfig,
    ReplayBuffer,
    train,
)

# Configuration
config = TrainingConfig(
    total_steps=100_000,
    batch_size=16,
    sequence_length=50,
    learning_rate=3e-4,
    grad_clip=100.0,
)

# Callbacks
from worldmodels.training.callbacks import (
    LoggingCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    ProgressCallback,
)
```

## Examples

See the `examples/` directory:

- `train_dreamer.py` - DreamerV3 training example
- `train_tdmpc2.py` - TD-MPC2 training example

```bash
# Quick test with random data
python examples/train_dreamer.py --test

# Train with real data
python examples/train_dreamer.py --data trajectories.npz --steps 100000
```

## Security

See [SECURITY.md](SECURITY.md) for security considerations, especially regarding loading model checkpoints from untrusted sources.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{worldmodels_sdk,
  title = {World Models SDK},
  year = {2024},
  url = {https://github.com/worldmodels/worldmodels-sdk}
}
```
