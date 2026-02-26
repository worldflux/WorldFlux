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

from .core import (
    ACTION_COMPONENTS_KEY,
    PLANNER_HORIZON_KEY,
    PLANNER_SEQUENCE_KEY,
    ActionConditioner,
    ActionPayload,
    ActionSequence,
    ActionSpec,
    AsyncDecoder,
    AsyncDynamicsModel,
    AsyncObservationEncoder,
    AsyncRolloutExecutor,
    AutoConfig,
    AutoWorldModel,
    Batch,
    BatchProvider,
    BatchProviderV2,
    BatchRequest,
    Capability,
    CategoricalLatentSpace,
    ComponentSpec,
    ConditionPayload,
    ConditionSpec,
    Decoder,
    DiffusionWorldModelConfig,
    DreamerV3Config,
    DynamicsModel,
    DynamicsType,
    GaussianLatentSpace,
    JEPABaseConfig,
    LatentSpace,
    LatentType,
    LossOutput,
    ModalityKind,
    ModalitySpec,
    ModelIOContract,
    ModelMaturity,
    ModelOutput,
    ObservationSpec,
    PredictionSpec,
    RolloutExecutor,
    SequenceFieldSpec,
    SequenceLayout,
    SequenceProvider,
    SimNormLatentSpace,
    State,
    StateSpec,
    TDMPC2Config,
    TokenProvider,
    TokenSpec,
    TokenWorldModelConfig,
    Trajectory,
    TransitionProvider,
    VideoProvider,
    VJEPA2Config,
    WorldModel,
    WorldModelConfig,
    WorldModelInput,
    WorldModelOutput,
    WorldModelRegistry,
    normalize_planned_action,
)
from .factory import (
    MODEL_ALIASES,
    MODEL_CATALOG,
    create_world_model,
    get_config,
    get_model_info,
    list_models,
)
from .models import (
    DiffusionWorldModel,
    DreamerV3WorldModel,
    JEPABaseWorldModel,
    TDMPC2WorldModel,
    TokenWorldModel,
    VJEPA2WorldModel,
)
from .planners import Planner, PlannerObjective, RewardObjective
from .utils import set_seed

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


# Lazy import for training module (optional dependency) + deprecation shim
def __getattr__(name: str) -> object:
    if name == "training":
        from . import training

        return training
    if name in _DEPRECATED_IMPORTS:
        import importlib
        import warnings

        mod_path, full_path = _DEPRECATED_IMPORTS[name]
        warnings.warn(
            f"Importing '{name}' from 'worldflux' is deprecated. "
            f"Use '{full_path}' instead. Will be removed in v0.2.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        mod = importlib.import_module(mod_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__) + list(_DEPRECATED_IMPORTS)


try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("worldflux")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

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
