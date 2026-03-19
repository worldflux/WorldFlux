# Error Handling Reference

WorldFlux uses a structured exception hierarchy so that callers can catch
errors at the right granularity. Every exception carries a machine-readable
`error_code` attribute (e.g. `WF-C001`) for logging and issue reports.

## Error Code Scheme

| Prefix | Category | Module |
|--------|----------|--------|
| WF-C | Configuration | `core.exceptions.ConfigurationError` |
| WF-S | Shape / Tensor | `core.exceptions.ShapeMismatchError` |
| WF-S1 | State | `core.exceptions.StateError` |
| WF-V | Validation | `core.exceptions.ValidationError` |
| WF-V0 | Contract | `core.exceptions.ContractValidationError` |
| WF-A | Capability | `core.exceptions.CapabilityError` |
| WF-M | Model Registry | `core.exceptions.ModelNotFoundError` |
| WF-K | Checkpoint | `core.exceptions.CheckpointError` |
| WF-T | Training | `core.exceptions.TrainingError` |
| WF-B | Buffer | `core.exceptions.BufferError` |

## Exception Hierarchy

```
WorldFluxError (WF-E000)
  +-- ConfigurationError (WF-C001)
  +-- ShapeMismatchError (WF-S001)
  +-- StateError (WF-S101)
  +-- ValidationError (WF-V001)
  |     +-- ContractValidationError (WF-V002)
  +-- CapabilityError (WF-A001)
  +-- ModelNotFoundError (WF-M001)
  +-- CheckpointError (WF-K001)
  +-- TrainingError (WF-T001)
  +-- BufferError (WF-B001)
```

---

## 1. ConfigurationError (WF-C001)

Raised when model configuration is invalid.

**Common causes:**
- Misspelled parameter name in `create_world_model()`
- Parameter value out of valid range
- Incompatible parameter combination

**Problem code:**
```python
from worldflux import create_world_model

# Typo: 'action_dims' instead of 'action_dim'
model = create_world_model("dreamerv3:size12m", action_dims=4)
# -> ConfigurationError: Unknown parameter 'action_dims' for DreamerV3Config.
#    Did you mean: 'action_dim'?
```

**Fix:**
```python
from worldflux import create_world_model

model = create_world_model("dreamerv3:size12m", action_dim=4)
```

**Catch pattern:**
```python
from worldflux.core.exceptions import ConfigurationError

try:
    model = create_world_model("dreamerv3:size12m", lerning_rate=1e-4)
except ConfigurationError as exc:
    print(f"[{exc.error_code}] {exc}")
    # WF-C001: Unknown parameter 'lerning_rate'. Did you mean: 'learning_rate'?
```

---

## 2. ShapeMismatchError (WF-S001)

Raised when tensor shapes do not match expected dimensions.

**Common causes:**
- Batch dimension mismatch between observations and actions
- Wrong channel ordering (HWC vs CHW)
- Encoder/decoder dimension incompatibility

**Problem code:**
```python
import torch

obs = torch.randn(8, 64, 64, 3)  # HWC format - wrong!
model.forward(obs, actions)
# -> ShapeMismatchError: expected (B, 3, 64, 64), got (8, 64, 64, 3)
```

**Fix:**
```python
obs = torch.randn(8, 3, 64, 64)  # CHW format - correct
model.forward(obs, actions)
```

---

## 3. StateError (WF-S101)

Raised when a State object is in an invalid state.

**Common causes:**
- Accessing latent state before calling `initial_state()`
- Shape mismatch after environment reset
- Attempting to modify a frozen state

**Problem code:**
```python
state = None
output = model.forward(obs, actions, state)
# -> StateError: state must be initialized first
```

**Fix:**
```python
state = model.initial_state(batch_size=8)
output = model.forward(obs, actions, state)
```

---

## 4. ValidationError (WF-V001)

Raised when runtime validation of tensors or values fails.

**Common causes:**
- NaN or Inf detected in intermediate computation
- Assertion failure in model forward pass
- Tensor value out of expected range

**Catch pattern:**
```python
from worldflux.core.exceptions import ValidationError

try:
    output = model.forward(obs, actions, state)
except ValidationError as exc:
    print(f"Validation failed: {exc}")
```

---

## 5. ContractValidationError (WF-V002)

Subclass of `ValidationError`. Raised when model I/O does not conform to the
declared `ModelIOContract`.

**Common causes:**
- Model output missing required fields (e.g. `next_state`)
- Output shape does not match the contract specification
- Incompatible modality types between input and contract

**Problem code:**
```python
# Custom model that forgot to return 'reward_pred'
class BadModel(WorldModel):
    def forward(self, obs, actions, state):
        return {"next_state": state}  # missing reward_pred
```

**Fix:**
```python
class FixedModel(WorldModel):
    def forward(self, obs, actions, state):
        return {"next_state": state, "reward_pred": reward}
```

---

## 6. CapabilityError (WF-A001)

Raised when requesting a capability the model does not support.

**Common causes:**
- Calling a planning method on a model without planning capability
- Requesting a component slot the model architecture does not expose
- Using a feature gated behind a config flag that is disabled

**Problem code:**
```python
# TD-MPC2 has planning; DreamerV3 uses imagination instead
model = create_world_model("dreamerv3:size12m")
model.plan(obs)  # -> CapabilityError if plan() is not supported
```

---

## 7. ModelNotFoundError (WF-M001)

Raised when a requested model identifier is not found in the registry.

**Common causes:**
- Misspelled model identifier
- Model plugin not installed
- Using a removed alias

**Problem code:**
```python
model = create_world_model("dreemrv3:size12m")  # typo
# -> ModelNotFoundError: Model 'dreemrv3:size12m' not found
```

**Fix:**
```python
from worldflux import list_models

print(list_models())  # see available identifiers
model = create_world_model("dreamerv3:size12m")
```

---

## 8. CheckpointError (WF-K001)

Raised when checkpoint loading or saving fails.

**Common causes:**
- Checkpoint file not found or corrupted
- Architecture mismatch between saved checkpoint and current model
- Missing keys in the state dictionary

**Problem code:**
```python
model = create_world_model("./missing_checkpoint/")
# -> CheckpointError: checkpoint file not found
```

**Fix:**
```python
import os
assert os.path.exists("./checkpoint/model.pt")
model = create_world_model("./checkpoint/")
```

---

## 9. TrainingError (WF-T001)

Raised when training encounters a fatal error.

**Common causes:**
- NaN detected in loss computation
- Gradient explosion despite clipping
- Out-of-memory during forward/backward pass

**Catch pattern:**
```python
from worldflux.core.exceptions import TrainingError

try:
    trainer.train(steps=10000)
except TrainingError as exc:
    print(f"Training stopped: [{exc.error_code}] {exc}")
```

---

## 10. BufferError (WF-B001)

Raised when replay buffer operations fail.

**Common causes:**
- Sampling from an empty buffer
- Shape mismatch when inserting transitions
- Concurrent write access (ReplayBuffer is NOT thread-safe - single writer
  thread only)

**Problem code:**
```python
buffer = ReplayBuffer(capacity=10000)
batch = buffer.sample(32)  # buffer is empty
# -> BufferError: cannot sample from empty buffer
```

**Fix:**
```python
buffer = ReplayBuffer(capacity=10000)
buffer.add(transition)  # add data first
batch = buffer.sample(32)
```

---

## General Catch Pattern

To catch all WorldFlux errors:

```python
from worldflux.core.exceptions import WorldFluxError

try:
    model = create_world_model("dreamerv3:size12m")
    output = model.forward(obs, actions, state)
except WorldFluxError as exc:
    print(f"WorldFlux error [{exc.error_code}]: {exc}")
```

To catch specific categories:

```python
from worldflux.core.exceptions import (
    ConfigurationError,
    ShapeMismatchError,
    TrainingError,
)

try:
    model = create_world_model("dreamerv3:size12m", **user_kwargs)
except ConfigurationError:
    print("Check your configuration parameters.")
except ShapeMismatchError:
    print("Check tensor shapes (obs_shape, action_dim).")
except TrainingError:
    print("Training diverged - try reducing learning_rate.")
```
