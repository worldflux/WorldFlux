# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Training run manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA_VERSION = "worldflux.training.run_manifest.v1"
CHECKPOINT_SCHEMA_VERSION = 1


def build_run_manifest(*, trainer: Any) -> dict[str, Any]:
    runtime_profile = trainer.runtime_profile() if hasattr(trainer, "runtime_profile") else {}
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "backend": str(getattr(trainer.config, "backend", "native_torch")),
        "device": str(getattr(trainer, "device", "cpu")),
        "global_step": int(trainer.state.global_step),
        "best_loss": float(trainer.state.best_loss),
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "training_config": trainer.config.to_dict(),
        "runtime_profile": runtime_profile,
        "train_start_time": trainer.state.train_start_time,
        "train_end_time": trainer.state.train_end_time,
        "ttfi_sec": trainer.state.ttfi_sec,
    }


def write_run_manifest(*, trainer: Any, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_run_manifest(trainer=trainer)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
