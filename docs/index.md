# WorldLoom

**Unified Interface for Latent World Models**

*One API. Multiple Architectures. Infinite Imagination.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yoshihyoda/worldloom/blob/main/examples/worldloom_quickstart.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-worldloom-blue?logo=github)](https://github.com/yoshihyoda/worldloom)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

WorldLoom provides a unified Python interface for latent world models used in reinforcement learning. Train and deploy DreamerV3, TD-MPC2, and more through a single, consistent API.

## Features

- **Unified API**: Common interface across DreamerV3, TD-MPC2, and more
- **Simple Usage**: One-liner model creation with `create_world_model()`
- **Training Infrastructure**: Complete training loop with callbacks, checkpointing, and logging
- **Type Safe**: Full type annotations and mypy compatibility

## Quick Start

```python
from worldloom import create_world_model
import torch

# Create a world model
model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(3, 64, 64),
    action_dim=4,
)

# Encode observation to latent state
obs = torch.randn(1, 3, 64, 64)
state = model.encode(obs)

# Imagine 15 steps into the future
actions = torch.randn(15, 1, 4)
trajectory = model.imagine(state, actions)

print(f"Predicted rewards: {trajectory.rewards.shape}")  # [15, 1, 1]
```

## Available Models

| Model | Best For | Presets |
|-------|----------|---------|
| **DreamerV3** | Images, Atari | `size12m`, `size25m`, `size50m`, `size100m`, `size200m` |
| **TD-MPC2** | State vectors, MuJoCo | `5m`, `19m`, `48m`, `317m` |

## Documentation

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install WorldLoom and learn the basics.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

    [:octicons-arrow-right-24: Core Concepts](getting-started/concepts.md)

-   **Tutorials**

    ---

    Step-by-step guides for common tasks.

    [:octicons-arrow-right-24: Train Your First Model](tutorials/train-first-model.md)

    [:octicons-arrow-right-24: DreamerV3 vs TD-MPC2](tutorials/dreamer-vs-tdmpc2.md)

-   **API Reference**

    ---

    Complete API documentation.

    [:octicons-arrow-right-24: Factory Functions](api/factory.md)

    [:octicons-arrow-right-24: WorldModel Protocol](api/protocol.md)

    [:octicons-arrow-right-24: Training](api/training.md)

</div>

## Architecture

```mermaid
graph LR
    subgraph Input
        A[Observation]
    end

    subgraph WorldModel["World Model"]
        B[Encoder]
        C[LatentState]
        D[Dynamics]
        E[Decoder]
    end

    subgraph Output
        F[Predictions]
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

## Installation

```bash
git clone https://github.com/yoshihyoda/worldloom.git
cd worldloom
pip install -e ".[training]"
```

## Try It Now

The fastest way to get started is our [interactive Colab notebook](https://colab.research.google.com/github/yoshihyoda/worldloom/blob/main/examples/worldloom_quickstart.ipynb).

## Contributing

Contributions are welcome! See our [Contributing Guide](https://github.com/yoshihyoda/worldloom/blob/main/CONTRIBUTING.md).

## License

MIT License - see [LICENSE](https://github.com/yoshihyoda/worldloom/blob/main/LICENSE) for details.
