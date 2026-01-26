"""
WorldLoom - Unified interface for latent world models.

Simple Usage:
    from worldloom import create_world_model

    # Create a DreamerV3 model
    model = create_world_model("dreamerv3:size12m")

    # Create a TD-MPC2 model with custom obs shape
    model = create_world_model("tdmpc2:5m", obs_shape=(39,), action_dim=4)

    # Use aliases for convenience
    model = create_world_model("dreamer")  # defaults to dreamerv3:size12m

Available Models:
    - dreamerv3:size12m, size25m, size50m, size100m, size200m
    - tdmpc2:5m, 19m, 48m, 317m

Aliases:
    - "dreamer", "dreamer-small", "dreamer-medium", "dreamer-large"
    - "tdmpc", "tdmpc-small", "tdmpc-medium", "tdmpc-large"
"""

from .core import (
    AutoConfig,
    AutoWorldModel,
    CategoricalLatentSpace,
    DreamerV3Config,
    DynamicsType,
    GaussianLatentSpace,
    LatentSpace,
    LatentState,
    LatentType,
    SimNormLatentSpace,
    TDMPC2Config,
    Trajectory,
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
from .models import DreamerV3WorldModel, TDMPC2WorldModel


# Lazy import for training module (optional dependency)
def __getattr__(name: str):
    if name == "training":
        from . import training

        return training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("worldloom")
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
    "LatentState",
    "Trajectory",
    "LatentType",
    "DynamicsType",
    "WorldModelConfig",
    "DreamerV3Config",
    "TDMPC2Config",
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
]
