# WorldFlux

<div align="center">

<img src="https://raw.githubusercontent.com/worldflux/WorldFlux/main/assets/logo.svg" alt="WorldFlux Logo" width="200">

**Unified Interface for World Models in Reinforcement Learning**

*One API. Multiple Architectures. Infinite Imagination.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/worldflux/Worldflux/blob/main/examples/worldflux_quickstart.ipynb)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/WorldFlux/demo)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-7289da?logo=discord&logoColor=white)](https://discord.gg/ZUBn9UEp2z)
[![PyPI](https://img.shields.io/pypi/v/worldflux.svg)](https://pypi.org/project/worldflux/)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

</div>

---

WorldFlux provides a unified Python interface for world models used in reinforcement learning. Starting with efficient latent-space models (DreamerV3, TD-MPC2), with plans to support diverse architectures including autoregressive and diffusion-based world models.

## Demo

### Imagination Rollout

World model rolls out future frames from a single observation:

![Imagination Rollout](https://raw.githubusercontent.com/worldflux/WorldFlux/main/docs/assets/imagination_rollout.gif)

*Left: Real game frames | Right: Model's imagination*

### Latent Space Dynamics

Visualization of how different episodes traverse the learned latent space:

![Latent Space](https://raw.githubusercontent.com/worldflux/WorldFlux/main/docs/assets/latent_space.gif)

## Features

- **Unified API**: Common interface across DreamerV3, TD-MPC2, and more
- **Simple Usage**: One-liner model creation with `create_world_model()`
- **Training Infrastructure**: Complete training loop with callbacks, checkpointing, and logging
- **Type Safe**: Full type annotations and mypy compatibility

## Model Maturity Policy

- **Reference**: DreamerV3, TD-MPC2 (release baseline)
- **Experimental**: JEPA, V-JEPA2, Token, Diffusion (API/metrics may evolve)

## Extension Policy

WorldFlux follows a **contract-first** extension policy:

- define `io_contract()` before adding model-specific training logic
- validate required batch/state keys at runtime
- promote families from experimental to reference only after gate compliance

You can inspect maturity tiers programmatically via:

```python
from worldflux import list_models

reference_models = list_models(maturity="reference")
experimental_models = list_models(maturity="experimental")
```

## Architecture

```mermaid
graph LR
    subgraph Input
        A[Observation]
    end

    subgraph WorldModel["World Model"]
        B[Encoder]
        C[State]
        D[Dynamics<br/>RSSM/MLP]
        E[Decoder]
    end

    subgraph Output
        F[Predictions<br/>obs, reward, continue]
    end

    A --> B
    B --> C
    C --> D
    D --> C
    C --> E
    E --> F

    style C fill:#e1f5fe
    style D fill:#fff3e0
```

**Key Concepts:**
- **Encoder**: Maps observations to latent states
- **State**: Compact representation (deterministic + stochastic components)
- **Dynamics**: Predicts next latent state from current state + action
- **Decoder**: Reconstructs observations/rewards from latent states

## Model Comparison

| Feature | DreamerV3 | TD-MPC2 |
|---------|-----------|---------|
| **Input Type** | Images or State | State vectors |
| **Latent Space** | Categorical (discrete) | SimNorm (continuous) |
| **Architecture** | RSSM | Implicit MLP |
| **Decoder** | Yes | No (implicit) |
| **Best For** | Atari, visual tasks | MuJoCo, robotics |
| **Planning** | Policy rollouts | MPC with Q-ensemble |

## Installation

### From Source (recommended)

```bash
git clone https://github.com/worldflux/Worldflux.git
cd worldflux
pip install -e "."

# With training dependencies
pip install -e ".[training]"

# With all optional dependencies
pip install -e ".[all]"

# For development
pip install -e ".[dev]"
```

### From PyPI

```bash
pip install worldflux
```

## Quick Start

### Create a Model

```python
from worldflux import create_world_model

# DreamerV3 (image observations)
model = create_world_model("dreamerv3:size12m")

# TD-MPC2 (state observations)
model = create_world_model("tdmpc2:5m", obs_shape=(39,), action_dim=6)
```

### Imagination Rollout

```python
import torch

# Encode initial observation
obs = torch.randn(1, 3, 64, 64)
state = model.encode(obs)

# Imagine 15 steps into the future
actions = torch.randn(15, 1, 4)  # [horizon, batch, action_dim]
trajectory = model.rollout(state, actions)

# Access predictions
print(f"Predicted rewards: {trajectory.rewards.shape}")
print(f"Continue probs: {trajectory.continues.shape}")
```

### Train a Model

```python
from worldflux import create_world_model
from worldflux.training import train, ReplayBuffer

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
from worldflux import create_world_model
from worldflux.training import Trainer, TrainingConfig, ReplayBuffer

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

### JEPA

| Preset | Parameters | Description |
|--------|------------|-------------|
| `jepa:base` | ~1M+ | Representation prediction |

### V-JEPA2

| Preset | Parameters | Description |
|--------|------------|-------------|
| `vjepa2:ci` | ~0.2M+ | CI smoke-test representation model |
| `vjepa2:tiny` | ~1M+ | Lightweight predictive representation |
| `vjepa2:base` | ~4M+ | Predictive representation baseline |

### Token

| Preset | Parameters | Description |
|--------|------------|-------------|
| `token:base` | ~1M+ | Discrete token dynamics |

### Diffusion

| Preset | Parameters | Description |
|--------|------------|-------------|
| `diffusion:base` | ~1M+ | Generative diffusion dynamics |

## API Reference

### Core Methods

All world models implement the `WorldModel` base class:

```python
# Encode observation to latent state
state = model.encode(obs)

# Predict next state (imagination, no observation)
next_state = model.transition(state, action)

# Update state with observation (posterior)
next_state = model.update(state, action, obs)

# Decode latent state to predictions
output = model.decode(state)
preds = output.preds  # e.g. {"obs", "reward", "continue"}

# Multi-step imagination rollout
trajectory = model.rollout(initial_state, actions)

# Compute training losses
loss_out = model.loss(batch)  # LossOutput (loss_out.loss, loss_out.components)
```

### Training API

```python
from worldflux.training import (
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
from worldflux.training.callbacks import (
    LoggingCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    ProgressCallback,
)
```

## Examples

See the `examples/` directory:

- `worldflux_quickstart.ipynb` - Interactive Colab notebook
- `train_dreamer.py` - DreamerV3 training example
- `train_tdmpc2.py` - TD-MPC2 training example
- `visualize_imagination.py` - Imagination rollout visualization

```bash
# Quick test with random data
python examples/train_dreamer.py --test

# Train with real data
python examples/train_dreamer.py --data trajectories.npz --steps 100000

# Collect Atari data
python examples/collect_atari.py --env Breakout --episodes 100

# Train on Atari
python examples/train_atari_dreamer.py --data atari_data.npz --steps 100000
```

## Benchmarks

Results on standard benchmarks:

| Environment | Model | Score | Training Steps |
|-------------|-------|-------|----------------|
| Atari Breakout | DreamerV3-50M | - | 200K |
| MuJoCo HalfCheetah | TD-MPC2-19M | - | 1M |
| DMControl Walker | DreamerV3-25M | - | 500K |

## Documentation

- [Full Documentation](https://github.com/worldflux/WorldFlux/tree/main/docs) - Comprehensive guides and API reference
- [Tutorials](https://github.com/worldflux/WorldFlux/tree/main/docs/tutorials) - Step-by-step learning
- [API Reference](https://github.com/worldflux/WorldFlux/tree/main/docs/api) - Detailed API docs

## Community

Join our [Discord](https://discord.gg/ZUBn9UEp2z) to discuss world models, get help, and connect with other researchers and developers.

## Security

See [SECURITY.md](SECURITY.md) for security considerations, especially regarding loading model checkpoints from untrusted sources.

## License

Apache License 2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) before submitting pull requests.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{worldflux,
  title = {WorldFlux: Unified Interface for World Models},
  year = {2026},
  url = {https://github.com/worldflux/Worldflux}
}
```

## Acknowledgments

WorldFlux builds on the excellent research from:
- [DreamerV3](https://arxiv.org/abs/2301.04104) - Hafner et al.
- [TD-MPC2](https://arxiv.org/abs/2310.16828) - Hansen et al.
