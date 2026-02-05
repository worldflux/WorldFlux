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
    - jepa:base, token:base, diffusion:base

Aliases:
    - "dreamer", "dreamer-small", "dreamer-medium", "dreamer-large"
    - "tdmpc", "tdmpc-small", "tdmpc-medium", "tdmpc-large"
    - "jepa", "token", "diffusion"
"""

from .core import (
    ActionSpec,
    AutoConfig,
    AutoWorldModel,
    Batch,
    BatchProvider,
    Capability,
    CategoricalLatentSpace,
    DiffusionWorldModelConfig,
    DreamerV3Config,
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
    WorldModel,
    WorldModelConfig,
    WorldModelRegistry,
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
)
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
    "BatchProvider",
    "TransitionProvider",
    "SequenceProvider",
    "TokenProvider",
    "VideoProvider",
    "ModelOutput",
    "LossOutput",
    # Specs
    "ModalityKind",
    "ModalitySpec",
    "ObservationSpec",
    "ActionSpec",
    "StateSpec",
    "TokenSpec",
    "PredictionSpec",
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
    "TokenWorldModel",
    "DiffusionWorldModel",
    # Utils
    "set_seed",
]
