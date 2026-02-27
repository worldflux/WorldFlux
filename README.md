# WorldFlux

<div align="center">

<img src="https://raw.githubusercontent.com/worldflux/WorldFlux/main/assets/logo.svg" alt="WorldFlux Logo" width="200">

**Unified Interface for World Models in Reinforcement Learning**

*One API. Multiple Architectures. Infinite Imagination.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/worldflux/WorldFlux/blob/main/examples/worldflux_quickstart.ipynb)
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

WorldFlux provides a unified Python interface for world models used in reinforcement learning.

## Demo

https://github.com/worldflux/WorldFlux/releases/download/demo-assets/dogrun.mp4

## Features

- **Unified API**: Common interface across model families
- **v3-first API**: `create_world_model()` defaults to `api_version="v3"` (strict contracts enabled)
- **Universal Payload Layer**: `ActionPayload` / `ConditionPayload` for polymorphic conditioning
- **Planner Contract**: planners return `ActionPayload` with `extras["wf.planner.horizon"]`
- **Simple Usage**: One-liner model creation with `create_world_model()`
- **Pluggable 5-layer core**: optional `component_overrides` for encoder/dynamics/conditioner/decoder/rollout
- **Training Infrastructure**: Complete training loop with callbacks, checkpointing, and logging
- **Type Safe**: Full type annotations and mypy compatibility

## Installation

Install `uv` first if you do not have it yet: [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Global CLI Install (cargo new style)

```bash
uv tool install worldflux
worldflux init my-world-model
```

Optional: enable the InquirerPy-powered prompt UI.

```bash
uv tool install --with inquirerpy worldflux
```

`worldflux init` now performs cross-platform pre-init dependency assurance.
It provisions a user-scoped bootstrap virtual environment and installs the
selected environment dependencies before scaffolding:

- Linux/macOS default: `~/.worldflux/bootstrap/py<major><minor>`
- Windows default: `%LOCALAPPDATA%/WorldFlux/bootstrap/py<major><minor>`

Environment variables:

- `WORLDFLUX_BOOTSTRAP_HOME`: override bootstrap root path
- `WORLDFLUX_INIT_ENSURE_DEPS=0`: disable auto-bootstrap (emergency bypass)

### From Source (recommended)

```bash
git clone https://github.com/worldflux/WorldFlux.git
cd worldflux
uv sync
source .venv/bin/activate
worldflux init my-world-model

# With training dependencies
uv sync --extra training

# With all optional dependencies
uv sync --extra all

# For development
uv sync --extra dev

# For documentation tooling
uv sync --extra docs
```

### From PyPI

```bash
uv pip install worldflux
worldflux init my-world-model
```

### Build Docs Locally

```bash
uv sync --extra docs
uv run mkdocs serve
uv run mkdocs build --strict
```

## Quick Start

### CPU-First Success Path (Official)

```bash
uv sync --extra dev
uv run python examples/quickstart_cpu_success.py --quick
```

### Create a Model

```python
from worldflux import create_world_model

model = create_world_model("dreamerv3:size12m")
```

### Universal Payload Usage (v3)

```python
from worldflux import ActionPayload, ConditionPayload

state = model.encode(obs)
next_state = model.transition(
    state,
    ActionPayload(kind="continuous", tensor=action),
    conditions=ConditionPayload(goal=goal_tensor),
)
```

### Component Overrides (5-layer core)

```python
from worldflux import create_world_model

model = create_world_model(
    "tdmpc2:ci",
    obs_shape=(4,),
    action_dim=2,
    component_overrides={
        # values can be registered component ids, classes, or instances
        "action_conditioner": "my_plugin.zero_action_conditioner",
    },
)
```

External packages can register plugins through entry-point groups:
- `worldflux.models`
- `worldflux.components`

### Imagination Rollout

```python
import torch

obs = torch.randn(1, 3, 64, 64)
state = model.encode(obs)

actions = torch.randn(15, 1, 4)  # [horizon, batch, action_dim]
trajectory = model.rollout(state, actions)

print(f"Predicted rewards: {trajectory.rewards.shape}")
print(f"Continue probs: {trajectory.continues.shape}")
```

### Train a Model

```python
from worldflux import create_world_model
from worldflux.training import train, ReplayBuffer

model = create_world_model("dreamerv3:size12m", obs_shape=(4,), action_dim=2)
buffer = ReplayBuffer.load("trajectories.npz")
trained_model = train(model, buffer, total_steps=50_000)
trained_model.save_pretrained("./my_model")
```

## Available Models

| Family | Presets |
|--------|---------|
| DreamerV3 | `size12m`, `size25m`, `size50m`, `size100m`, `size200m` |
| TD-MPC2 | `5m`, `19m`, `48m`, `317m` |
| JEPA | `base` |
| V-JEPA2 | `ci`, `tiny`, `base` |
| Token | `base` |
| Diffusion | `base` |

## API Reference

### Core Methods

All world models implement the `WorldModel` base class:

```python
state = model.encode(obs)
next_state = model.transition(state, action)
next_state = model.update(state, action, obs)
output = model.decode(state)
preds = output.preds  # e.g. {"obs", "reward", "continue"}
trajectory = model.rollout(initial_state, actions)
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

from worldflux.training.callbacks import (
    LoggingCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
    ProgressCallback,
)
```

## Examples

See the `examples/` directory:

- `quickstart_cpu_success.py` - Official CPU-first success path
- `compare_unified_training.py` - Same trainer/data contract for DreamerV3 and TD-MPC2
- `worldflux_quickstart.ipynb` - Interactive Colab notebook
- `train_dreamer.py` - Training example
- `train_tdmpc2.py` - Training example
- `visualize_imagination.py` - Imagination rollout visualization

```bash
uv run python examples/quickstart_cpu_success.py --quick
uv run python examples/compare_unified_training.py --quick
uv run python examples/train_dreamer.py --test
uv run python examples/train_dreamer.py --data trajectories.npz --steps 100000
```

## Documentation

- [Full Documentation](https://worldflux.ai/) - Guides and API reference
- [API Reference](https://worldflux.ai/api/factory/) - Contract and symbol-level docs
- [Reference](https://worldflux.ai/reference/benchmarks/) - Operational and quality docs

## Community

Join our [Discord](https://discord.gg/ZUBn9UEp2z) to discuss world models, get help, and connect with other researchers and developers.

- Support channels and response paths: [SUPPORT.md](SUPPORT.md)
- Community expectations and reporting: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Security

See [SECURITY.md](SECURITY.md) for security considerations, especially regarding loading model checkpoints from untrusted sources.

## License

Apache License 2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

## Contributing

Contributions are welcome. Please read our [Contributing Guide](CONTRIBUTING.md) before submitting pull requests.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{worldflux,
  title = {WorldFlux: Unified Interface for World Models},
  year = {2026},
  url = {https://github.com/worldflux/WorldFlux}
}
```
