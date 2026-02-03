"""Core components for world models."""

from .batch import Batch, BatchProvider
from .config import (
    DreamerV3Config,
    DynamicsType,
    JEPABaseConfig,
    LatentType,
    TDMPC2Config,
    WorldModelConfig,
)
from .exceptions import (
    BufferError,
    CheckpointError,
    ConfigurationError,
    ModelNotFoundError,
    ShapeMismatchError,
    StateError,
    TrainingError,
    WorldFluxError,
)
from .latent_space import (
    CategoricalLatentSpace,
    GaussianLatentSpace,
    LatentSpace,
    SimNormLatentSpace,
)
from .model import WorldModel
from .output import LossOutput, ModelOutput
from .registry import AutoConfig, AutoWorldModel, WorldModelRegistry
from .spec import (
    ActionSpec,
    Capability,
    ModalityKind,
    ModalitySpec,
    ObservationSpec,
    StateSpec,
    TokenSpec,
)
from .state import State
from .trajectory import Trajectory

__all__ = [
    # State and Trajectory
    "State",
    "Trajectory",
    "Batch",
    "BatchProvider",
    "ModelOutput",
    "LossOutput",
    # Specs
    "ModalityKind",
    "ModalitySpec",
    "ObservationSpec",
    "ActionSpec",
    "StateSpec",
    "TokenSpec",
    "Capability",
    # Config
    "LatentType",
    "DynamicsType",
    "WorldModelConfig",
    "DreamerV3Config",
    "TDMPC2Config",
    "JEPABaseConfig",
    # Protocol and Registry
    "WorldModel",
    "WorldModelRegistry",
    "AutoWorldModel",
    "AutoConfig",
    # Latent Spaces
    "LatentSpace",
    "GaussianLatentSpace",
    "CategoricalLatentSpace",
    "SimNormLatentSpace",
    # Exceptions
    "WorldFluxError",
    "ConfigurationError",
    "ShapeMismatchError",
    "StateError",
    "ModelNotFoundError",
    "CheckpointError",
    "TrainingError",
    "BufferError",
]
