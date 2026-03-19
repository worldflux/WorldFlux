# Migration Guide: v0.2 to v3

This guide covers the breaking changes between the v0.2 (legacy) API and the
current v3 API, along with concrete before/after code examples and a
deprecation timeline.

## Deprecation Timeline

| Milestone | Date (estimated) | Behavior |
|-----------|-------------------|----------|
| v0.2.0 | 2026-Q1 (current) | `DeprecationWarning` emitted when `api_version="v0.2"` |
| v0.3.0 | 2026-Q3 | `DeprecationWarning` intensified; `api_version` defaults to `"v3"` |
| v0.4.0 | 2027-Q1 | v0.2 API fully removed; `api_version="v0.2"` raises `ValueError` |

---

## Summary of Breaking Changes

1. **`api_version` parameter** - v0.2 adapters are deprecated.
2. **`action_type="hybrid"` removed** in v3.
3. **`obs_shape` format** - v3 always expects CHW tuples.
4. **`observation_modalities`** replaces single-modality assumption.
5. **`action_spec`** replaces `action_type` + `action_dim` pattern.
6. **`kwargs` are now validated** - misspelled params raise `ConfigurationError`.
7. **Skeleton configs** removed from top-level `__init__.py`.
8. **`RolloutEngine`** renamed to `RolloutExecutor`.
9. **`first_action`** helper removed from public API.
10. **`PluginManifest`** moved to `worldflux.core.registry`.
11. **Builder pattern** added as alternative to `create_world_model()`.

---

## Before/After Examples

### 1. Basic model creation (api_version)

**Before (v0.2):**
```python
from worldflux import create_world_model

model = create_world_model("dreamerv3:size12m", api_version="v0.2")
```

**After (v3):**
```python
from worldflux import create_world_model

model = create_world_model("dreamerv3:size12m")
# api_version defaults to "v3", no need to specify
```

### 2. Action type specification

**Before (v0.2):**
```python
model = create_world_model(
    "tdmpc2:5m",
    obs_shape=(39,),
    action_dim=6,
    action_type="continuous",
)
```

**After (v3):**
```python
model = create_world_model(
    "tdmpc2:5m",
    obs_shape=(39,),
    action_dim=6,
    action_spec={"kind": "continuous", "dim": 6},
)
```

### 3. Hybrid actions (removed)

**Before (v0.2):**
```python
model = create_world_model(
    "dreamerv3:size12m",
    action_type="hybrid",
    api_version="v0.2",
)
```

**After (v3):**
```python
# Hybrid actions are not supported in v3.
# Use separate continuous/discrete action specs instead:
model = create_world_model(
    "dreamerv3:size12m",
    action_spec={"kind": "continuous", "dim": 4},
)
```

### 4. Observation shape (HWC to CHW)

**Before (v0.2):**
```python
# Some v0.2 code used HWC ordering
model = create_world_model("dreamerv3:size12m", obs_shape=(64, 64, 3))
```

**After (v3):**
```python
# v3 expects CHW (PyTorch convention)
model = create_world_model("dreamerv3:size12m", obs_shape=(3, 64, 64))
```

### 5. Multi-modal observations

**Before (v0.2):**
```python
# v0.2 only supported a single obs_shape
model = create_world_model("dreamerv3:size12m", obs_shape=(3, 64, 64))
```

**After (v3):**
```python
model = create_world_model(
    "dreamerv3:size12m",
    observation_modalities={
        "image": {"kind": "image", "shape": (3, 64, 64)},
        "proprio": {"kind": "vector", "shape": (7,)},
    },
)
```

### 6. Skeleton config imports

**Before (v0.2):**
```python
from worldflux import DiTSkeletonConfig
```

**After (v3):**
```python
from worldflux.core.config import DiTSkeletonConfig
# Top-level import emits DeprecationWarning and will be removed in v0.4.0
```

### 7. RolloutEngine renamed

**Before (v0.2):**
```python
from worldflux import RolloutEngine
```

**After (v3):**
```python
from worldflux import RolloutExecutor
# Or use the async variant:
from worldflux import AsyncRolloutExecutor
```

### 8. PluginManifest moved

**Before (v0.2):**
```python
from worldflux import PluginManifest
```

**After (v3):**
```python
from worldflux.core.registry import PluginManifest
```

### 9. first_action helper removed

**Before (v0.2):**
```python
from worldflux import first_action
action = first_action(action_sequence)
```

**After (v3):**
```python
from worldflux.core.payloads import first_action
action = first_action(action_sequence)
```

### 10. Using the Builder pattern (new in v3)

**Before (v0.2):**
```python
model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(3, 64, 64),
    action_dim=6,
    device="cuda",
    learning_rate=1e-4,
)
```

**After (v3) - Builder alternative:**
```python
from worldflux import WorldModelBuilder

model = (
    WorldModelBuilder("dreamerv3:size12m")
    .with_obs_shape((3, 64, 64))
    .with_action_dim(6)
    .with_device("cuda")
    .with_config(learning_rate=1e-4)
    .build()
)
```

### 11. Kwargs are now validated

**Before (v0.2):**
```python
# Typo silently ignored - 'lerning_rate' had no effect
model = create_world_model("dreamerv3:size12m", lerning_rate=1e-4)
```

**After (v3):**
```python
# Typo raises ConfigurationError with suggestion
model = create_world_model("dreamerv3:size12m", lerning_rate=1e-4)
# -> ConfigurationError: Unknown parameter 'lerning_rate' for DreamerV3Config.
#    Did you mean: 'learning_rate'?
```

### 12. Config introspection via CLI (new in v3)

```bash
# Inspect all fields of a config preset
worldflux config inspect dreamerv3:size12m

# Compare two presets
worldflux config diff dreamerv3:size12m dreamerv3:size200m

# Validate a config file
worldflux config validate ./my_config.json

# Generate JSON Schema
worldflux config schema dreamerv3:size12m
```

---

## Automated Migration Script

A basic migration script is provided at `scripts/migrate_v02_to_v3.py`.
It handles the most common textual transformations:

```bash
# Dry run (show changes without writing)
python scripts/migrate_v02_to_v3.py --dry-run path/to/your_code.py

# Apply changes in-place
python scripts/migrate_v02_to_v3.py path/to/your_code.py
```

The script performs the following replacements:
- `api_version="v0.2"` - removed (v3 is default)
- `from worldflux import RolloutEngine` - updated to `RolloutExecutor`
- `from worldflux import PluginManifest` - updated to full path
- `from worldflux import first_action` - updated to full path
- Skeleton config imports updated to `worldflux.core.config`
- `action_type="hybrid"` - flagged for manual review

**Note:** The script handles textual replacements only. Complex refactors
(e.g. switching from `action_type` to `action_spec` dicts) require manual
review.
