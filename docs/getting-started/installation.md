# Installation

## Requirements

- Python 3.10+
- PyTorch 2.0+

## From Source (Recommended)

```bash
git clone https://github.com/worldloom/WorldLoom.git
cd worldloom
pip install -e "."
```

### With Optional Dependencies

```bash
# Training infrastructure (Trainer, ReplayBuffer, callbacks)
pip install -e ".[training]"

# Visualization (matplotlib, imageio for GIFs)
pip install -e ".[viz]"

# Atari environments (gymnasium[atari], ale-py)
pip install -e ".[atari]"

# All optional dependencies
pip install -e ".[all]"

# Development (testing, linting, type checking)
pip install -e ".[dev]"
```

## From PyPI (Coming Soon)

```bash
pip install worldloom
```

## Verify Installation

```python
import worldloom
print(worldloom.__version__)

from worldloom import create_world_model, list_models
print(list_models())
```

## GPU Support

WorldLoom automatically uses CUDA if available:

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
pip install -e ".[training]"
```
