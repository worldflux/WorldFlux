# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## From Source (Recommended)

```bash
git clone https://github.com/worldflux/Worldflux.git
cd worldflux
uv sync
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

## From PyPI (Coming Soon)

```bash
uv pip install worldflux
```

## Verify Installation

```python
import worldflux
print(worldflux.__version__)

from worldflux import create_world_model, list_models
print(list_models())
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
