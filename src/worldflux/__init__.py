"""
WorldFlux - Unified interface for latent world models.

Simple Usage:
    from worldflux import create_world_model

    # Create a DreamerV3 model
    model = create_world_model("dreamerv3:size12m")

    # Create a TD-MPC2 model with custom obs shape
    model = create_world_model("tdmpc2:5m", obs_shape=(39,), action_dim=4)

    # Use aliases for convenience
    model = create_world_model("dreamer")  # defaults to dreamerv3:size12m

Available Models:
    - dreamerv3:size12m, size25m, size50m, size100m, size200m
    - tdmpc2:5m, 19m, 48m, 317m
    - jepa:base, vjepa2:ci, vjepa2:tiny, vjepa2:base, token:base, diffusion:base

Aliases:
    - "dreamer", "dreamer-small", "dreamer-medium", "dreamer-large"
    - "tdmpc", "tdmpc-small", "tdmpc-medium", "tdmpc-large"
    - "jepa", "vjepa2", "token", "diffusion"
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, tuple[str, ...]] = {
    "worldflux.factory": (
        "MODEL_ALIASES",
        "MODEL_CATALOG",
        "create_world_model",
        "get_config",
        "get_model_info",
        "list_models",
    ),
    "worldflux.core.batch": (
        "Batch",
        "BatchProvider",
        "BatchProviderV2",
        "BatchRequest",
        "SequenceProvider",
        "TokenProvider",
        "TransitionProvider",
        "VideoProvider",
    ),
    "worldflux.core.output": ("LossOutput", "ModelOutput", "WorldModelOutput"),
    "worldflux.core.payloads": (
        "ACTION_COMPONENTS_KEY",
        "PLANNER_HORIZON_KEY",
        "PLANNER_SEQUENCE_KEY",
        "ActionPayload",
        "ActionSequence",
        "ConditionPayload",
        "WorldModelInput",
        "normalize_planned_action",
    ),
    "worldflux.core.interfaces": (
        "ActionConditioner",
        "AsyncDecoder",
        "AsyncDynamicsModel",
        "AsyncObservationEncoder",
        "AsyncRolloutExecutor",
        "ComponentSpec",
        "Decoder",
        "DynamicsModel",
        "RolloutExecutor",
    ),
    "worldflux.planners": ("Planner", "PlannerObjective", "RewardObjective"),
    "worldflux.core.spec": (
        "ActionSpec",
        "Capability",
        "ConditionSpec",
        "ModalityKind",
        "ModalitySpec",
        "ModelIOContract",
        "ModelMaturity",
        "ObservationSpec",
        "PredictionSpec",
        "SequenceFieldSpec",
        "SequenceLayout",
        "StateSpec",
        "TokenSpec",
    ),
    "worldflux.core.config": (
        "DiffusionWorldModelConfig",
        "DreamerV3Config",
        "DynamicsType",
        "JEPABaseConfig",
        "LatentType",
        "TDMPC2Config",
        "TokenWorldModelConfig",
        "VJEPA2Config",
        "WorldModelConfig",
    ),
    "worldflux.core.model": ("WorldModel",),
    "worldflux.core.registry": ("AutoConfig", "AutoWorldModel", "WorldModelRegistry"),
    "worldflux.core.latent_space": (
        "CategoricalLatentSpace",
        "GaussianLatentSpace",
        "LatentSpace",
        "SimNormLatentSpace",
    ),
    "worldflux.core.state": ("State",),
    "worldflux.core.trajectory": ("Trajectory",),
    "worldflux.models.dreamer": ("DreamerV3WorldModel",),
    "worldflux.models.tdmpc2": ("TDMPC2WorldModel",),
    "worldflux.models.jepa": ("JEPABaseWorldModel",),
    "worldflux.models.vjepa2": ("VJEPA2WorldModel",),
    "worldflux.models.token": ("TokenWorldModel",),
    "worldflux.models.diffusion": ("DiffusionWorldModel",),
    "worldflux.utils": ("set_seed",),
}
_EXPORTS: dict[str, str] = {
    name: module_path for module_path, names in _EXPORT_GROUPS.items() for name in names
}

# Deprecated symbols: (source_module, fully_qualified_path)
_DEPRECATED_IMPORTS: dict[str, tuple[str, str]] = {
    # Registry / interfaces / payloads (removed from top-level __all__)
    "PluginManifest": ("worldflux.core.registry", "worldflux.core.registry.PluginManifest"),
    "RolloutEngine": ("worldflux.core.interfaces", "worldflux.core.interfaces.RolloutEngine"),
    "first_action": ("worldflux.core.payloads", "worldflux.core.payloads.first_action"),
    "is_namespaced_extra_key": (
        "worldflux.core.payloads",
        "worldflux.core.payloads.is_namespaced_extra_key",
    ),
    # Skeleton configs (non-public, use worldflux.core.config directly)
    "DiTSkeletonConfig": ("worldflux.core.config", "worldflux.core.config.DiTSkeletonConfig"),
    "SSMSkeletonConfig": ("worldflux.core.config", "worldflux.core.config.SSMSkeletonConfig"),
    "Renderer3DSkeletonConfig": (
        "worldflux.core.config",
        "worldflux.core.config.Renderer3DSkeletonConfig",
    ),
    "PhysicsSkeletonConfig": (
        "worldflux.core.config",
        "worldflux.core.config.PhysicsSkeletonConfig",
    ),
    "GANSkeletonConfig": ("worldflux.core.config", "worldflux.core.config.GANSkeletonConfig"),
    # Skeleton world models (non-public, use worldflux.models.<name> directly)
    "DiTSkeletonWorldModel": (
        "worldflux.models.dit",
        "worldflux.models.dit.DiTSkeletonWorldModel",
    ),
    "SSMSkeletonWorldModel": (
        "worldflux.models.ssm",
        "worldflux.models.ssm.SSMSkeletonWorldModel",
    ),
    "Renderer3DSkeletonWorldModel": (
        "worldflux.models.renderer3d",
        "worldflux.models.renderer3d.Renderer3DSkeletonWorldModel",
    ),
    "PhysicsSkeletonWorldModel": (
        "worldflux.models.physics",
        "worldflux.models.physics.PhysicsSkeletonWorldModel",
    ),
    "GANSkeletonWorldModel": (
        "worldflux.models.gan",
        "worldflux.models.gan.GANSkeletonWorldModel",
    ),
}


def _load_export(name: str) -> Any:
    module_path = _EXPORTS[name]
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    if name == "training":
        module = import_module("worldflux.training")
        globals()[name] = module
        return module
    if name in _EXPORTS:
        return _load_export(name)
    if name in _DEPRECATED_IMPORTS:
        import warnings

        mod_path, full_path = _DEPRECATED_IMPORTS[name]
        warnings.warn(
            f"Importing '{name}' from 'worldflux' is deprecated. "
            f"Use '{full_path}' instead. Will be removed in v0.2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        module = import_module(mod_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_DEPRECATED_IMPORTS) | {"training"})


try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("worldflux")
except PackageNotFoundError:
    __version__ = "0.1.1.dev0"

__all__ = [
    # Simple API (recommended)
    "create_world_model",
    "list_models",
    "get_model_info",
    "get_config",
    "MODEL_ALIASES",
    "MODEL_CATALOG",
    # Core
    "State",
    "Trajectory",
    "Batch",
    "BatchRequest",
    "BatchProvider",
    "BatchProviderV2",
    "TransitionProvider",
    "SequenceProvider",
    "TokenProvider",
    "VideoProvider",
    "ModelOutput",
    "WorldModelOutput",
    "LossOutput",
    "ActionPayload",
    "ActionSequence",
    "ConditionPayload",
    "WorldModelInput",
    "normalize_planned_action",
    "PLANNER_HORIZON_KEY",
    "PLANNER_SEQUENCE_KEY",
    "ACTION_COMPONENTS_KEY",
    "ComponentSpec",
    "ActionConditioner",
    "AsyncObservationEncoder",
    "AsyncDynamicsModel",
    "AsyncDecoder",
    "AsyncRolloutExecutor",
    "DynamicsModel",
    "Decoder",
    "RolloutExecutor",
    "Planner",
    "PlannerObjective",
    "RewardObjective",
    # Specs
    "ModalityKind",
    "ModalitySpec",
    "ObservationSpec",
    "ActionSpec",
    "ConditionSpec",
    "StateSpec",
    "TokenSpec",
    "PredictionSpec",
    "SequenceFieldSpec",
    "SequenceLayout",
    "ModelIOContract",
    "Capability",
    "ModelMaturity",
    "LatentType",
    "DynamicsType",
    "WorldModelConfig",
    "DreamerV3Config",
    "TDMPC2Config",
    "JEPABaseConfig",
    "VJEPA2Config",
    "TokenWorldModelConfig",
    "DiffusionWorldModelConfig",
    "WorldModel",
    "WorldModelRegistry",
    "AutoWorldModel",
    "AutoConfig",
    # Latent spaces
    "LatentSpace",
    "GaussianLatentSpace",
    "CategoricalLatentSpace",
    "SimNormLatentSpace",
    # Models
    "DreamerV3WorldModel",
    "TDMPC2WorldModel",
    "JEPABaseWorldModel",
    "VJEPA2WorldModel",
    "TokenWorldModel",
    "DiffusionWorldModel",
    # Utils
    "set_seed",
]
