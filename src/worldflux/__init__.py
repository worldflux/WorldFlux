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
    DiTSkeletonConfig,
    DreamerV3Config,
    DynamicsModel,
    DynamicsType,
    GANSkeletonConfig,
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
    PhysicsSkeletonConfig,
    PluginManifest,
    PredictionSpec,
    Renderer3DSkeletonConfig,
    RolloutEngine,
    RolloutExecutor,
    SequenceFieldSpec,
    SequenceLayout,
    SequenceProvider,
    SimNormLatentSpace,
    SSMSkeletonConfig,
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
    first_action,
    is_namespaced_extra_key,
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
    DiTSkeletonWorldModel,
    DreamerV3WorldModel,
    GANSkeletonWorldModel,
    JEPABaseWorldModel,
    PhysicsSkeletonWorldModel,
    Renderer3DSkeletonWorldModel,
    SSMSkeletonWorldModel,
    TDMPC2WorldModel,
    TokenWorldModel,
    VJEPA2WorldModel,
)
from .planners import Planner, PlannerObjective, RewardObjective
from .utils import set_seed


# Lazy import for training module (optional dependency)
def __getattr__(name: str):
    if name == "training":
        from . import training

        return training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "first_action",
    "PLANNER_HORIZON_KEY",
    "PLANNER_SEQUENCE_KEY",
    "ACTION_COMPONENTS_KEY",
    "is_namespaced_extra_key",
    "ComponentSpec",
    "ActionConditioner",
    "AsyncObservationEncoder",
    "AsyncDynamicsModel",
    "AsyncDecoder",
    "AsyncRolloutExecutor",
    "DynamicsModel",
    "Decoder",
    "RolloutExecutor",
    "RolloutEngine",
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
    "DiTSkeletonConfig",
    "SSMSkeletonConfig",
    "Renderer3DSkeletonConfig",
    "PhysicsSkeletonConfig",
    "GANSkeletonConfig",
    "WorldModel",
    "WorldModelRegistry",
    "PluginManifest",
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
    "DiTSkeletonWorldModel",
    "SSMSkeletonWorldModel",
    "Renderer3DSkeletonWorldModel",
    "PhysicsSkeletonWorldModel",
    "GANSkeletonWorldModel",
    # Utils
    "set_seed",
]
