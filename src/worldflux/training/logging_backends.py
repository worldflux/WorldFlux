# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Pluggable logging backend system for training metrics."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LoggingBackend(Protocol):
    """Protocol for training metric logging backends."""

    def log_scalar(self, tag: str, value: float, step: int) -> None: ...

    def log_histogram(self, tag: str, values: Any, step: int) -> None: ...

    def log_image(self, tag: str, image: Any, step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class CSVBackend:
    """Logs scalars to a local CSV file."""

    def __init__(self, log_dir: str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._rows.append({"step": step, "tag": tag, "value": value})

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        pass  # CSV does not support histograms

    def log_image(self, tag: str, image: Any, step: int) -> None:
        pass  # CSV does not support images

    def flush(self) -> None:
        if not self._rows:
            return
        csv_path = self._log_dir / "scalars.csv"
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "tag", "value"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()


class TensorBoardBackend:
    """Logs metrics to TensorBoard via SummaryWriter."""

    def __init__(self, log_dir: str) -> None:
        self._writer: Any = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=log_dir)
        except ImportError:
            logger.warning("tensorboard not installed, TensorBoardBackend disabled")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        if self._writer:
            self._writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        if self._writer:
            self._writer.add_image(tag, image, step)

    def flush(self) -> None:
        if self._writer:
            self._writer.flush()

    def close(self) -> None:
        if self._writer:
            self._writer.close()


class WandbBackend:
    """Logs metrics to Weights & Biases."""

    def __init__(
        self,
        project: str = "worldflux",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._run: Any = None
        try:
            import wandb

            self._run = wandb.init(project=project, name=run_name, config=config)
        except ImportError:
            logger.warning("wandb not installed, WandbBackend disabled")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self._run:
            import wandb

            wandb.log({tag: value, "step": step})

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        if self._run:
            import wandb

            wandb.log({tag: wandb.Histogram(values), "step": step})

    def log_image(self, tag: str, image: Any, step: int) -> None:
        if self._run:
            import wandb

            wandb.log({tag: wandb.Image(image), "step": step})

    def flush(self) -> None:
        pass  # wandb auto-flushes

    def close(self) -> None:
        if self._run:
            self._run.finish()


class CompositeBackend:
    """Routes log calls to multiple backends simultaneously."""

    def __init__(self, backends: list[Any]) -> None:
        self._backends = backends

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        for b in self._backends:
            b.log_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        for b in self._backends:
            b.log_histogram(tag, values, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        for b in self._backends:
            b.log_image(tag, image, step)

    def flush(self) -> None:
        for b in self._backends:
            b.flush()

    def close(self) -> None:
        for b in self._backends:
            b.close()
