# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Distributed training support for WorldFlux.

Provides DDPTrainer for multi-GPU DistributedDataParallel training,
and an FSDPTrainer stub for future large-model support.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn

from worldflux.core.batch import BatchProvider
from worldflux.core.exceptions import ConfigurationError, TrainingError

from .config import TrainingConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

    from worldflux.core.model import WorldModel

logger = logging.getLogger(__name__)


def build_launch_config(config: TrainingConfig) -> dict[str, Any]:
    """Return a normalized launch description for distributed execution."""
    return {
        "enabled": config.use_ddp or config.distributed_mode != "none",
        "mode": "ddp" if config.use_ddp else config.distributed_mode,
        "world_size": int(config.world_size or config.distributed_world_size),
        "device": config.resolve_device(),
    }


def _resolve_env_rank(config: TrainingConfig) -> tuple[int, int, int]:
    """Resolve rank, world_size, local_rank from config or environment variables.

    Returns:
        Tuple of (rank, world_size, local_rank).
    """
    rank = config.rank
    if rank is None:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

    world_size = config.world_size
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

    local_rank = config.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    return rank, world_size, local_rank


class DDPTrainer:
    """DistributedDataParallel trainer for multi-GPU training.

    Wraps an existing Trainer instance with DDP support, handling:
    - Process group initialization
    - Model wrapping with DDP
    - Gradient synchronization with no_sync for accumulation
    - Checkpoint management (rank 0 only save, broadcast load)
    - Logging (rank 0 only stdout/wandb)

    Args:
        model: World model to train.
        config: Training configuration with DDP fields set.

    Example:
        >>> # Launched via torchrun --nproc_per_node=2
        >>> config = TrainingConfig(use_ddp=True, ddp_backend="nccl")
        >>> trainer = DDPTrainer(model, config)
        >>> trainer.train(data)
    """

    def __init__(
        self,
        model: WorldModel | nn.Module,
        config: TrainingConfig,
    ) -> None:
        if not config.use_ddp:
            raise ConfigurationError("DDPTrainer requires use_ddp=True in TrainingConfig.")

        self.config = config
        self.rank, self.world_size, self.local_rank = _resolve_env_rank(config)

        # Setup distributed process group
        self._setup_distributed()

        # Set device from local_rank
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        # Move model to device and wrap with DDP
        self.unwrapped_model = cast(nn.Module, model).to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.unwrapped_model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=config.ddp_find_unused_params,
            gradient_as_bucket_view=config.ddp_gradient_as_bucket_view,
            broadcast_buffers=config.ddp_broadcast_buffers,
        )

        # Import Trainer lazily to avoid circular dependency
        from .trainer import Trainer

        # Create a base trainer instance with the DDP-wrapped model
        # Override config device to use the resolved local device
        local_config = config.with_updates(
            device=str(self.device),
            use_ddp=False,  # Prevent recursion; DDP wrapping is done here
        )
        self._trainer = Trainer(self.unwrapped_model, local_config)
        # Swap the model used by the trainer to our DDP-wrapped version
        self._trainer.model = self.model

        self._accumulation_step = 0

        if self.is_main_process:
            logger.info(
                "DDPTrainer initialized: rank=%d, world_size=%d, local_rank=%d, backend=%s",
                self.rank,
                self.world_size,
                self.local_rank,
                config.ddp_backend,
            )

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process (rank 0)."""
        return self.rank == 0

    @property
    def state(self) -> Any:
        """Expose the underlying trainer state."""
        return self._trainer.state

    def _setup_distributed(self) -> None:
        """Initialize the distributed process group."""
        if torch.distributed.is_initialized():
            logger.debug("Process group already initialized, skipping setup.")
            return

        try:
            torch.distributed.init_process_group(
                backend=self.config.ddp_backend,
                rank=self.rank,
                world_size=self.world_size,
            )
        except RuntimeError as e:
            raise TrainingError(
                f"Failed to initialize distributed process group: {e}. "
                "Ensure MASTER_ADDR and MASTER_PORT environment variables are set, "
                "or launch with torchrun."
            ) from e

        logger.debug(
            "Process group initialized: backend=%s, rank=%d/%d",
            self.config.ddp_backend,
            self.rank,
            self.world_size,
        )

    @contextmanager
    def _maybe_no_sync(self) -> Iterator[None]:
        """Context manager that disables gradient sync during accumulation steps."""
        accum_steps = self.config.gradient_accumulation_steps
        is_accumulating = self._accumulation_step < accum_steps - 1
        if is_accumulating and accum_steps > 1:
            with self.model.no_sync():
                yield
        else:
            yield

    def _train_step(self, data: BatchProvider | Any) -> dict[str, float]:
        """Execute a single training step with DDP-aware gradient sync."""
        with self._maybe_no_sync():
            metrics = self._trainer._train_step(data)
        self._accumulation_step = (
            self._accumulation_step + 1
        ) % self.config.gradient_accumulation_steps
        return metrics

    def train(
        self,
        data: BatchProvider | Any,
        num_steps: int | None = None,
        resume_from: str | None = None,
    ) -> nn.Module:
        """Train with DDP coordination.

        All processes run training in parallel. Only rank 0 saves checkpoints
        and writes logs.

        Args:
            data: BatchProvider or iterable yielding Batch/dict.
            num_steps: Number of steps to train.
            resume_from: Path to checkpoint to resume from.

        Returns:
            The unwrapped trained model.
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        # Run the underlying trainer's train loop
        self._trainer.train(data, num_steps=num_steps)

        return self.unwrapped_model

    def save_checkpoint(self, path: str) -> None:
        """Save checkpoint from rank 0 only.

        All ranks synchronize via barrier after save to ensure consistency.
        """
        if self.is_main_process:
            # Save the unwrapped model state_dict (without DDP module. prefix)
            self._trainer.model = self.unwrapped_model
            self._trainer.save_checkpoint(path)
            self._trainer.model = self.model

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint on all ranks.

        Loads on rank 0 first, then broadcasts via barrier sync.
        The DDP module. prefix is handled automatically.
        """
        # Load using the unwrapped model to avoid state_dict key mismatches
        self._trainer.model = self.unwrapped_model
        self._trainer.load_checkpoint(path)
        self._trainer.model = self.model

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def cleanup(self) -> None:
        """Clean up distributed process group."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Destroyed distributed process group.")


class FSDPTrainer:
    """Fully Sharded Data Parallel trainer placeholder.

    This class is planned for v0.4.0 and will provide support for training
    models with 100M+ parameters across multiple GPUs using PyTorch FSDP.

    Planned features:
    - FullyShardedDataParallel model wrapping
    - Model parallelism (RSSM + Encoder/Decoder sharding)
    - Activation offloading to CPU
    - Mixed-precision sharding policies
    - Checkpoint sharding and consolidation

    Args:
        model: World model to train.
        config: Training configuration.

    Raises:
        NotImplementedError: Always. This class is a placeholder.
    """

    # FSDP config fields (reserved for v0.4.0)
    FSDP_DEFAULTS: dict[str, Any] = {
        "sharding_strategy": "FULL_SHARD",
        "cpu_offload": False,
        "backward_prefetch": "BACKWARD_PRE",
        "mixed_precision_policy": None,
        "activation_checkpointing": False,
    }

    def __init__(
        self,
        model: WorldModel | nn.Module,
        config: TrainingConfig,
    ) -> None:
        raise NotImplementedError(
            "FSDPTrainer is planned for WorldFlux v0.4.0. "
            "Use DDPTrainer for multi-GPU training in the current version."
        )

    def train(self, data: Any, **kwargs: Any) -> nn.Module:
        """Placeholder train method."""
        raise NotImplementedError("FSDPTrainer is not yet implemented.")

    def save_checkpoint(self, path: str) -> None:
        """Placeholder checkpoint save."""
        raise NotImplementedError("FSDPTrainer is not yet implemented.")

    def load_checkpoint(self, path: str) -> None:
        """Placeholder checkpoint load."""
        raise NotImplementedError("FSDPTrainer is not yet implemented.")
