# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Training run manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUN_MANIFEST_SCHEMA_VERSION = "worldflux.training.run_manifest.v1"
CHECKPOINT_SCHEMA_VERSION = 1


def _infer_support_surface(*, trainer: Any) -> str:
    raw_explicit = getattr(trainer, "support_surface", "")
    explicit = str(raw_explicit).strip().lower() if raw_explicit is not None else ""
    if explicit:
        return explicit

    model = getattr(trainer, "model", None)
    model_config = getattr(model, "config", None)
    model_type = str(getattr(model_config, "model_type", "")).strip().lower()
    backend = str(getattr(trainer.config, "backend", "native_torch")).strip().lower()

    if model_type in {"dreamer", "tdmpc2"}:
        return "advanced" if backend != "native_torch" else "supported"
    return "internal"


def _manifest_data_mode(*, trainer: Any) -> str:
    raw_value = getattr(trainer, "data_mode", "")
    value = str(raw_value).strip().lower() if raw_value is not None else ""
    return value or "unknown"


def _manifest_degraded_modes(*, trainer: Any) -> list[str]:
    raw = getattr(trainer, "degraded_modes", [])
    if not isinstance(raw, list | tuple):
        return []
    normalized: list[str] = []
    for item in raw:
        value = str(item).strip().lower()
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _infer_run_classification(*, trainer: Any, support_surface: str, data_mode: str) -> str:
    raw_explicit = getattr(trainer, "run_classification", "")
    explicit = str(raw_explicit).strip().lower() if raw_explicit is not None else ""
    if explicit:
        return explicit

    degraded = _manifest_degraded_modes(trainer=trainer)
    if support_surface == "advanced":
        return "advanced_evidence"
    if support_surface == "supported" and data_mode in {"offline", "online"} and not degraded:
        return "meaningful_local_training"
    return "contract_smoke"


def build_run_manifest(*, trainer: Any) -> dict[str, Any]:
    runtime_profile = trainer.runtime_profile() if hasattr(trainer, "runtime_profile") else {}
    support_surface = _infer_support_surface(trainer=trainer)
    data_mode = _manifest_data_mode(trainer=trainer)
    degraded_modes = _manifest_degraded_modes(trainer=trainer)
    payload = {
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
        "support_surface": support_surface,
        "data_mode": data_mode,
        "degraded_modes": degraded_modes,
        "run_classification": _infer_run_classification(
            trainer=trainer,
            support_surface=support_surface,
            data_mode=data_mode,
        ),
    }
    data_provenance = getattr(trainer, "data_provenance", None)
    if isinstance(data_provenance, dict) and data_provenance:
        payload["data_provenance"] = dict(data_provenance)
    return payload


def write_run_manifest(*, trainer: Any, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_run_manifest(trainer=trainer)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
