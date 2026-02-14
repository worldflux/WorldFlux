# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Global CLI Install (cargo new style)

```bash
uv tool install worldflux
worldflux init my-world-model
```

Optional: install InquirerPy for enhanced prompt widgets.

```bash
uv tool install --with inquirerpy worldflux
```

`worldflux init` performs pre-init dependency assurance on Linux/macOS/Windows.
Before generating files, it creates/uses a user-scoped bootstrap virtual environment
and installs dependencies for the selected environment:

- Linux/macOS default: `~/.worldflux/bootstrap/py<major><minor>`
- Windows default: `%LOCALAPPDATA%/WorldFlux/bootstrap/py<major><minor>`

You can override behavior with:

- `WORLDFLUX_BOOTSTRAP_HOME`: override bootstrap root directory
- `WORLDFLUX_INIT_ENSURE_DEPS=0`: disable auto-bootstrap (emergency bypass)

## From Source (Recommended)

```bash
git clone https://github.com/worldflux/WorldFlux.git
cd worldflux
uv sync
source .venv/bin/activate
worldflux init my-world-model
```

### With Optional Dependencies

```bash
# Training infrastructure (Trainer, ReplayBuffer, callbacks)
uv sync --extra training

# Visualization (matplotlib, imageio for GIFs)
uv sync --extra viz

# Atari environments (gymnasium[atari], ale-py)
uv sync --extra atari

# All optional dependencies
uv sync --extra all

# Development (testing, linting, type checking)
uv sync --extra dev

# Documentation tooling (MkDocs + API autogen plugins)
uv sync --extra docs
```

## From PyPI

```bash
uv pip install worldflux
worldflux init my-world-model
```

## Verify Installation

```python
import worldflux
print(worldflux.__version__)

from worldflux import create_world_model, list_models
print(list_models())
```

## CPU Success Check

```bash
uv sync --extra dev
uv run python examples/quickstart_cpu_success.py --quick
```

## GPU Support

WorldFlux automatically uses CUDA if available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# Models automatically use GPU when available
model = create_world_model("dreamerv3:size12m", device="cuda")
```

## Troubleshooting

### CUDA Out of Memory

Use smaller model presets or reduce batch size:

```python
# Use smaller model
model = create_world_model("dreamerv3:size12m")  # Instead of size200m

# Reduce training batch size
config = TrainingConfig(batch_size=8)  # Instead of 16
```

### Missing Dependencies

If you get import errors for training features:

```bash
uv sync --extra training
```

## Build Documentation Locally

```bash
uv sync --extra docs
uv run mkdocs serve
uv run mkdocs build --strict
```
