# Troubleshooting

Common issues and solutions.

## Installation Issues

### Import Error: No module named 'worldflux'

```
ModuleNotFoundError: No module named 'worldflux'
```

**Solution**: Install WorldFlux in development mode:

```bash
cd worldflux
pip install -e "."
```

### Missing training dependencies

```
ModuleNotFoundError: No module named 'worldflux.training'
```

**Solution**: Install with training extras:

```bash
pip install -e ".[training]"
```

### CUDA not available

```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solution**:
1. Check NVIDIA drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

---

## Memory Issues

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:

1. Use smaller model:
```python
model = create_world_model("dreamerv3:size12m")  # Instead of size200m
```

2. Reduce batch size:
```python
config = TrainingConfig(batch_size=8)  # Instead of 16
```

3. Reduce sequence length:
```python
config = TrainingConfig(sequence_length=25)  # Instead of 50
```

4. Enable gradient checkpointing:
```python
model.enable_gradient_checkpointing()
```

5. Use CPU for smaller experiments:
```python
model = create_world_model("dreamerv3:size12m", device="cpu")
```

### Memory grows during training

**Cause**: Tensors accumulating in memory.

**Solution**: Ensure losses are detached for logging:

```python
# Bad
metrics["loss"] = loss

# Good
metrics["loss"] = loss.item()
```

---

## Training Issues

### Loss is NaN

**Causes**:
- Learning rate too high
- Gradient explosion
- Bad data (NaN/Inf values)

**Solutions**:

1. Reduce learning rate:
```python
config = TrainingConfig(learning_rate=1e-4)
```

2. Enable gradient clipping:
```python
config = TrainingConfig(grad_clip=100.0)  # Default
```

3. Check data for NaN:
```python
import numpy as np
data = np.load("data.npz")
for key in data:
    assert not np.isnan(data[key]).any(), f"NaN in {key}"
    assert not np.isinf(data[key]).any(), f"Inf in {key}"
```

### Loss not decreasing

**Causes**:
- Insufficient data
- Learning rate too low
- Model too small

**Solutions**:

1. Collect more data (100+ episodes)
2. Increase learning rate:
```python
config = TrainingConfig(learning_rate=1e-3)
```
3. Use larger model:
```python
model = create_world_model("dreamerv3:size50m")
```
4. Train longer:
```python
config = TrainingConfig(total_steps=200_000)
```

### KL loss dominates (DreamerV3)

**Solution**: Adjust KL balancing:

```python
model = create_world_model(
    "dreamerv3:size50m",
    kl_free=1.0,      # Allow some free nats
    kl_balance=0.8,   # Balance prior/posterior
)
```

---

## Model Issues

### TD-MPC2: stochastic is None

This is expected. TD-MPC2 is an implicit model without stochastic state:

```python
state = tdmpc.encode(obs)
print(state.tensors["latent"])  # SimNorm embedding

# Use latent for features
features = state.tensors["latent"]
```

### TD-MPC2: trajectory.continues is None

This is expected. TD-MPC2 doesn't predict episode continuation:

```python
trajectory = tdmpc.rollout(state, actions)
print(trajectory.continues)  # None - this is normal!
```

### DreamerV3: Blurry reconstructions

**Causes**:
- Model too small
- Insufficient training
- KL too high

**Solutions**:
1. Use larger model
2. Train longer
3. Reduce KL weight:
```python
model = create_world_model("dreamerv3:size50m", kl_balance=0.5)
```

---

## Data Issues

### ReplayBuffer sample error

```
ValueError: Not enough data to sample
```

**Cause**: Buffer has fewer transitions than `batch_size * seq_len`.

**Solution**:
```python
# Check buffer size
print(f"Buffer size: {len(buffer)}")
print(f"Required: {batch_size * seq_len}")

# Collect more data or reduce batch/seq
batch = buffer.sample(batch_size=4, seq_len=10)
```

### Episode boundary issues

**Symptom**: Model learns wrong transitions at episode boundaries.

**Solution**: Ensure proper `dones` array:

```python
buffer.add_episode(
    obs=obs_array,
    actions=action_array,
    rewards=reward_array,
    dones=done_array,  # True at episode end, False elsewhere
)
```

---

## Loading Issues

### Can't load saved model

```
FileNotFoundError: config.json not found
```

**Solution**: Check model directory structure:

```
my_model/
├── config.json    # Must exist
└── model.pt       # Must exist
```

### Version mismatch

```
KeyError: 'new_parameter'
```

**Cause**: Model saved with different WorldFlux version.

**Solution**:
```python
# Load with strict=False
import torch
state_dict = torch.load("model.pt")
model.load_state_dict(state_dict, strict=False)
```

---

## Performance Issues

### Training too slow

**Solutions**:

1. Use GPU:
```python
model = create_world_model(..., device="cuda")
```

2. Increase batch size (if memory allows):
```python
config = TrainingConfig(batch_size=64)
```

3. Use DataLoader workers:
```python
# In custom training loop
dataloader = DataLoader(dataset, num_workers=4)
```

### Imagination too slow

**Solutions**:

1. Use torch.no_grad():
```python
with torch.no_grad():
    trajectory = model.rollout(state, actions)
```

2. Batch multiple rollouts:
```python
# Instead of: multiple single rollouts
# Do: one batched rollout
states = model.encode(obs_batch)  # [B, ...]
trajectory = model.rollout(states, actions)  # Batched
```

---

## Getting Help

If your issue isn't listed here:

1. Check [GitHub Issues](https://github.com/worldflux/Worldflux/issues)
2. Search existing issues for similar problems
3. Open a new issue with:
   - WorldFlux version (`worldflux.__version__`)
   - Python version
   - PyTorch version
   - Full error traceback
   - Minimal reproduction code
