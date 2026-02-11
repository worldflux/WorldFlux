"""Small visualization helpers for CPU-safe examples."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_reward_heatmap_ppm(
    rewards: np.ndarray,
    output_path: str | Path,
    *,
    height: int = 24,
) -> Path:
    """Write a simple reward heatmap to a PPM image file."""
    arr = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise ValueError("rewards must not be empty")

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    scale = max(hi - lo, 1e-8)
    norm = (arr - lo) / scale

    width = int(arr.shape[0])
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    # blue->red gradient so low/high values are visually separable without extra deps.
    canvas[..., 0] = (norm * 255.0).astype(np.uint8)
    canvas[..., 2] = ((1.0 - norm) * 255.0).astype(np.uint8)
    canvas[..., 1] = 32

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(canvas.tobytes())
    return path
