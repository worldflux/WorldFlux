"""Training configuration for WorldFlux."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from worldflux.core.exceptions import ConfigurationError


@dataclass
class TrainingConfig:
    """
    Configuration for training world models.

    Args:
        total_steps: Total number of training steps.
        batch_size: Batch size for training.
        sequence_length: Sequence length for trajectory sampling.
        learning_rate: Learning rate for optimizer.
        grad_clip: Maximum gradient norm for clipping.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        log_interval: Interval (in steps) for logging metrics.
        eval_interval: Interval (in steps) for evaluation.
        save_interval: Interval (in steps) for saving checkpoints.
        output_dir: Directory for saving outputs (checkpoints, logs).
        device: Device to train on ('cuda', 'cpu', 'auto').
        seed: Random seed for reproducibility.
        mixed_precision: Whether to use mixed precision training.
        num_workers: Number of workers for data loading.
        prefetch_factor: Number of batches to prefetch per worker.

    Example:
        >>> config = TrainingConfig(total_steps=100_000, batch_size=32)
        >>> config.save("training_config.json")
        >>> loaded = TrainingConfig.load("training_config.json")
    """

    # Training duration
    total_steps: int = 100_000
    batch_size: int = 16
    sequence_length: int = 50
    instant_mode: bool = False
    instant_total_steps: int = 8
    instant_batch_size: int = 8
    instant_sequence_length: int = 10

    # Optimizer
    learning_rate: float = 3e-4
    grad_clip: float = 100.0
    weight_decay: float = 0.0
    warmup_steps: int = 0

    # Logging and checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 10000
    output_dir: str = "./outputs"

    # Hardware
    device: str = "auto"
    seed: int = 42
    mixed_precision: bool = False
    mixed_precision_dtype: str = "float16"  # "float16" | "bfloat16"
    backend: str = "native_torch"
    backend_profile: str = ""
    distributed_mode: str = "none"
    distributed_world_size: int = 1

    # DDP (DistributedDataParallel) settings
    use_ddp: bool = False
    ddp_backend: str = "nccl"  # nccl (GPU) / gloo (CPU)
    ddp_find_unused_params: bool = False
    ddp_gradient_as_bucket_view: bool = True
    ddp_broadcast_buffers: bool = True
    rank: int | None = None
    world_size: int | None = None
    local_rank: int | None = None

    # Data loading
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_batches: int = 2

    # Vectorized environments
    num_envs: int = 1
    env_mode: str = "auto"  # "sync" | "async" | "auto"

    # Advanced options
    optimizer: str = "adamw"
    scheduler: str = "none"
    ema_decay: float | None = None

    # Gradient accumulation
    gradient_accumulation_steps: int = 1

    # Gradient checkpointing
    use_gradient_checkpointing: bool = False
    checkpoint_layers: list[str] = field(default_factory=list)

    # Quality check after training
    auto_quality_check: bool = True

    # torch.compile support (PyTorch 2.0+)
    compile_model: bool = False

    # Profiling
    profile: bool = False

    # Reserved for future trainer-level model patching. Currently unsupported by Trainer.
    model_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization.

        Raises:
            ConfigurationError: If any configuration values are invalid.
        """
        self._validate()

    def _validate(self) -> None:
        """Validate all configuration values.

        Raises:
            ConfigurationError: If any configuration values are invalid.
        """
        if self.total_steps <= 0:
            raise ConfigurationError(f"total_steps must be positive, got {self.total_steps}")
        if self.batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {self.batch_size}")
        if self.sequence_length <= 0:
            raise ConfigurationError(
                f"sequence_length must be positive, got {self.sequence_length}"
            )
        if self.instant_total_steps <= 0:
            raise ConfigurationError(
                f"instant_total_steps must be positive, got {self.instant_total_steps}"
            )
        if self.instant_batch_size <= 0:
            raise ConfigurationError(
                f"instant_batch_size must be positive, got {self.instant_batch_size}"
            )
        if self.instant_sequence_length <= 0:
            raise ConfigurationError(
                f"instant_sequence_length must be positive, got {self.instant_sequence_length}"
            )
        if self.learning_rate <= 0:
            raise ConfigurationError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.grad_clip < 0:
            raise ConfigurationError(f"grad_clip must be non-negative, got {self.grad_clip}")
        if self.weight_decay < 0:
            raise ConfigurationError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.warmup_steps < 0:
            raise ConfigurationError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
        if self.num_workers < 0:
            raise ConfigurationError(f"num_workers must be non-negative, got {self.num_workers}")
        if self.optimizer not in ("adamw", "adam", "sgd"):
            raise ConfigurationError(
                f"optimizer must be 'adamw', 'adam', or 'sgd', got '{self.optimizer}'"
            )
        if self.scheduler not in ("none", "linear", "cosine", "constant"):
            raise ConfigurationError(
                f"scheduler must be 'none', 'linear', 'cosine', or 'constant', "
                f"got '{self.scheduler}'"
            )
        if self.ema_decay is not None and not (0.0 < self.ema_decay < 1.0):
            raise ConfigurationError(f"ema_decay must be in (0, 1), got {self.ema_decay}")
        if self.gradient_accumulation_steps < 1:
            raise ConfigurationError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if not str(self.backend).strip():
            raise ConfigurationError("backend must not be empty")
        if self.distributed_mode not in ("none", "ddp"):
            raise ConfigurationError(
                f"distributed_mode must be 'none' or 'ddp', got {self.distributed_mode!r}"
            )
        if self.distributed_world_size < 1:
            raise ConfigurationError(
                f"distributed_world_size must be >= 1, got {self.distributed_world_size}"
            )
        if self.ddp_backend not in ("nccl", "gloo"):
            raise ConfigurationError(
                f"ddp_backend must be 'nccl' or 'gloo', got {self.ddp_backend!r}"
            )
        if self.mixed_precision_dtype not in ("float16", "bfloat16"):
            raise ConfigurationError(
                f"mixed_precision_dtype must be 'float16' or 'bfloat16', "
                f"got {self.mixed_precision_dtype!r}"
            )
        if self.num_envs < 1:
            raise ConfigurationError(f"num_envs must be >= 1, got {self.num_envs}")
        if self.env_mode not in ("sync", "async", "auto"):
            raise ConfigurationError(
                f"env_mode must be 'sync', 'async', or 'auto', got {self.env_mode!r}"
            )
        if self.prefetch_batches < 0:
            raise ConfigurationError(
                f"prefetch_batches must be non-negative, got {self.prefetch_batches}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        """Create config from dictionary."""
        return cls(**d)

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        """Load config from JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d)

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device.

        Priority: CUDA > MPS (Apple Silicon) > CPU.
        """
        if self.device == "auto":
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def with_updates(self, **kwargs: Any) -> TrainingConfig:
        """Return a new config with updated values.

        Args:
            **kwargs: Configuration values to update.

        Returns:
            New TrainingConfig with updated values.

        Raises:
            ConfigurationError: If updated values are invalid.
        """
        d = self.to_dict()
        d.update(kwargs)
        return TrainingConfig.from_dict(d)

    def effective_total_steps(self) -> int:
        """Total steps used by trainer under current mode."""
        if self.instant_mode:
            return self.instant_total_steps
        return self.total_steps

    def effective_batch_size(self) -> int:
        """Batch size used by trainer under current mode."""
        if self.instant_mode:
            return self.instant_batch_size
        return self.batch_size

    def effective_sequence_length(self) -> int:
        """Sequence length used by trainer under current mode."""
        if self.instant_mode:
            return self.instant_sequence_length
        return self.sequence_length

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"TrainingConfig("
            f"total_steps={self.total_steps}, "
            f"batch_size={self.batch_size}, "
            f"seq_len={self.sequence_length}, "
            f"instant_mode={self.instant_mode}, "
            f"lr={self.learning_rate}, "
            f"device={self.device!r}, "
            f"backend={self.backend!r})"
        )
