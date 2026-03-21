# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Dataset manifest helpers for reproducible evaluation and training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_MANIFEST_SCHEMA_VERSION = "worldflux.dataset.manifest.v1"


def build_dataset_manifest(
    *,
    env_id: str,
    collector_kind: str,
    collector_policy: str,
    seed: int,
    episodes: int,
    transitions: int,
    reward_stats: dict[str, float],
    source_commit: str,
    created_at: str,
    preprocessing: dict[str, Any],
    artifact_paths: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "env_id": env_id,
        "collector_kind": collector_kind,
        "collector_policy": collector_policy,
        "seed": int(seed),
        "episodes": int(episodes),
        "transitions": int(transitions),
        "reward_stats": dict(reward_stats),
        "source_commit": source_commit,
        "created_at": created_at,
        "preprocessing": dict(preprocessing),
        "artifact_paths": dict(artifact_paths),
    }


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset manifest must contain a JSON object: {manifest_path}")
    schema_version = str(payload.get("schema_version", "")).strip()
    if schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset manifest schema_version {schema_version!r}: {manifest_path}"
        )
    if not str(payload.get("env_id", "")).strip():
        raise ValueError(f"Dataset manifest missing env_id: {manifest_path}")
    return payload


def write_dataset_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def resolve_dataset_artifact_path(
    manifest: dict[str, Any],
    *,
    artifact_key: str,
    manifest_path: str | Path,
) -> Path:
    artifact_paths = manifest.get("artifact_paths", {})
    if not isinstance(artifact_paths, dict):
        raise ValueError("Dataset manifest artifact_paths must be an object.")

    raw_path = str(artifact_paths.get(artifact_key, "")).strip()
    if not raw_path:
        raise ValueError(f"Dataset manifest missing artifact path for {artifact_key!r}.")

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(manifest_path).expanduser().resolve().parent / path
