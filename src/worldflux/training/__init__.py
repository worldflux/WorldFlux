"""Training infrastructure for WorldFlux."""

from .backend import JobHandle, JobStatus, LocalBackend, TrainingBackend
from .callbacks import (
    Callback,
    CheckpointCallback,
    DiagnosisCallback,
    HeartbeatCallback,
    LoggingCallback,
    TrainingReportCallback,
)
from .config import TrainingConfig
from .data import ReplayBuffer, TokenSequenceProvider, TrajectoryDataset, TransitionArrayProvider
from .report import HealthSignal, LossCurveSummary, TrainingReport
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
    "TrainingReportCallback",
    # Report
    "TrainingReport",
    "HealthSignal",
    "LossCurveSummary",
]
