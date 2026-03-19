"""Performance profiling and monitoring for WorldFlux training.

Provides fine-grained timing instrumentation for each training phase
(batch loading, forward, backward, optimizer step) and GPU memory tracking.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


def _percentile(values: list[float], pct: float) -> float:
    """Compute percentile from a sorted or unsorted list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


class PerformanceMonitor:
    """Training performance monitor with phase-level timing and GPU metrics.

    Records per-step timing for each named phase and provides summary
    statistics (mean, p95, max).

    Args:
        enabled: Whether to actually record metrics. When False, the
            measure() context manager is a no-op for zero overhead.

    Example:
        >>> monitor = PerformanceMonitor(enabled=True)
        >>> with monitor.measure("forward_ms"):
        ...     output = model(batch)
        >>> with monitor.measure("backward_ms"):
        ...     loss.backward()
        >>> print(monitor.summary())
    """

    # Standard metric names
    METRIC_NAMES = (
        "batch_load_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_step_ms",
        "gpu_memory_peak_mb",
        "gpu_utilization_pct",
    )

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.metrics: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, metric_name: str) -> Any:
        """Context manager to measure elapsed time for a named phase.

        Records elapsed time in milliseconds under the given metric name.
        When the monitor is disabled, this is a no-op.

        Args:
            metric_name: Name of the metric to record (e.g. "forward_ms").
        """
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.metrics[metric_name].append(elapsed_ms)

    def record(self, metric_name: str, value: float) -> None:
        """Manually record a single metric value.

        Args:
            metric_name: Name of the metric.
            value: Value to record.
        """
        if not self.enabled:
            return
        self.metrics[metric_name].append(value)

    def record_gpu_memory(self) -> None:
        """Record current GPU peak memory usage (CUDA only).

        Tracks torch.cuda.max_memory_allocated() in megabytes. Safe to call
        on non-CUDA devices (silently skipped).
        """
        if not self.enabled:
            return
        try:
            import torch

            if torch.cuda.is_available():
                peak_bytes = torch.cuda.max_memory_allocated()
                peak_mb = peak_bytes / (1024 * 1024)
                self.metrics["gpu_memory_peak_mb"].append(peak_mb)
                # Reset peak tracker for next step
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def summary(self) -> dict[str, dict[str, float]]:
        """Compute summary statistics for all recorded metrics.

        Returns:
            Dict mapping metric_name to {mean, p95, max} statistics.
        """
        result: dict[str, dict[str, float]] = {}
        for name, values in self.metrics.items():
            if not values:
                continue
            result[name] = {
                "mean": sum(values) / len(values),
                "p95": _percentile(values, 95),
                "max": max(values),
                "count": float(len(values)),
            }
        return result

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()

    def log_summary(self) -> None:
        """Log a human-readable summary of all metrics."""
        if not self.enabled:
            return
        stats = self.summary()
        if not stats:
            logger.info("PerformanceMonitor: no metrics recorded.")
            return
        lines = ["Performance summary:"]
        for name, s in sorted(stats.items()):
            lines.append(f"  {name}: mean={s['mean']:.2f}, p95={s['p95']:.2f}, max={s['max']:.2f}")
        logger.info("\n".join(lines))


class TorchProfilerWrapper:
    """Wrapper for torch.profiler to produce Chrome trace output.

    Creates a torch.profiler.profile context that records CPU/GPU operations
    and exports a Chrome-compatible trace file.

    Args:
        output_dir: Directory for the trace file.
        wait_steps: Steps to wait before profiling starts.
        warmup_steps: Steps for profiler warmup.
        active_steps: Steps to actively profile.
        repeat: Number of profiling cycles.

    Example:
        >>> profiler = TorchProfilerWrapper("./profiles")
        >>> with profiler.profile() as p:
        ...     for step in range(100):
        ...         train_step()
        ...         p.step()
    """

    def __init__(
        self,
        output_dir: str = "./profiles",
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 3,
        repeat: int = 1,
    ) -> None:
        self.output_dir = output_dir
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat = repeat

    @contextmanager
    def profile(self) -> Any:
        """Context manager that yields a torch.profiler.profile instance.

        The trace is automatically exported to self.output_dir on exit.
        """
        import os

        import torch

        os.makedirs(self.output_dir, exist_ok=True)

        schedule = torch.profiler.schedule(
            wait=self.wait_steps,
            warmup=self.warmup_steps,
            active=self.active_steps,
            repeat=self.repeat,
        )

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        trace_handler = torch.profiler.tensorboard_trace_handler(self.output_dir)

        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,
            with_stack=True,
        ) as p:
            yield p

        logger.info("Profiler trace saved to %s", self.output_dir)
