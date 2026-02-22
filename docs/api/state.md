# State

The core representation used by all world models.

## Overview

`State` is a lightweight container with two fields:

- `tensors`: `dict[str, Tensor]` holding model-specific latent tensors
- `meta`: `dict[str, Any]` for optional metadata

```python
from worldflux.core.state import State
```

## Creating a State

Most users get a `State` via `model.encode()` or `model.update()`:

```python
state = model.encode(obs)
```

## Accessing Tensors

`State` does not fix a schema. Each model defines its own tensor keys.

### DreamerV3

- `deter`: deterministic GRU state
- `stoch`: stochastic categorical samples
- `prior_logits` / `posterior_logits`: logits for KL

```python
features = torch.cat(
    [state.tensors["deter"], state.tensors["stoch"].flatten(1)],
    dim=-1,
)
```

### TD-MPC2

- `latent`: SimNorm embedding

```python
latent = state.tensors["latent"]
```

### JEPA

- `rep`: encoder representation

```python
rep = state.tensors["rep"]
```

## Metadata

Use `meta` for non-tensor bookkeeping:

```python
state.meta["latent_type"] = "simnorm"
```

## State Operations

### Device Transfer

```python
# Move state to GPU
gpu_state = state.to("cuda")

# Move back to CPU
cpu_state = gpu_state.to("cpu")
```

### Detach and Clone

```python
# Detach from computation graph (e.g., for rollout targets)
detached = state.detach()

# Deep copy with independent tensors
cloned = state.clone()
```

### Validation

```python
# Verify all tensors have consistent batch dimension
state.validate()  # raises StateError if inconsistent
```

### Batch Size and Device Inspection

```python
print(state.batch_size)  # e.g., 32
print(state.device)      # e.g., device(type='cuda', index=0)
```

### Safe Tensor Access

```python
# Returns None instead of KeyError if key is missing
latent = state.get("latent")
deter = state.get("deter", default=torch.zeros(1, 256))
```

## Serialization

State supports binary serialization for checkpointing and IPC:

```python
# Serialize to bytes
data = state.serialize(version="v1", format="binary")

# Deserialize from bytes
restored = State.deserialize(data)
```

### Shared Memory (Zero-Copy IPC)

For multi-process training pipelines:

```python
# Producer process
descriptor = state.to_shared_memory(namespace="my-state")

# Consumer process
attached = State.from_shared_memory(descriptor, copy=False)

# Clean up
attached.close_shared_memory(unlink=True)
```

## Implementation

```python
@dataclass
class State:
    tensors: dict[str, Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
```

## API Reference

::: worldflux.core.state.State
    options:
      show_source: false
      members:
        - get
        - batch_size
        - device
        - to
        - detach
        - clone
        - validate
        - serialize
        - deserialize
        - to_shared_memory
        - from_shared_memory
        - close_shared_memory
