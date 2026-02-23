"""Training infrastructure for WorldFlux."""

from .backend import JobHandle, JobStatus, LocalBackend, TrainingBackend
from .callbacks import (
    Callback,
    CheckpointCallback,
    DiagnosisCallback,
    HeartbeatCallback,
    LoggingCallback,
)
from .config import TrainingConfig
from .data import ReplayBuffer, TokenSequenceProvider, TrajectoryDataset, TransitionArrayProvider
from .trainer import Trainer, train

__all__ = [
    # Main API
    "train",
    "Trainer",
    "TrainingConfig",
    "ReplayBuffer",
    "TrajectoryDataset",
    "TokenSequenceProvider",
    "TransitionArrayProvider",
    "TrainingBackend",
    "LocalBackend",
    "JobHandle",
    "JobStatus",
    # Callbacks
    "Callback",
    "LoggingCallback",
    "CheckpointCallback",
    "HeartbeatCallback",
    "DiagnosisCallback",
]
