# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for dataset manifest helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldflux.training.dataset_manifest import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    build_dataset_manifest,
    load_dataset_manifest,
    resolve_dataset_artifact_path,
    write_dataset_manifest,
)


def test_write_and_load_dataset_manifest(tmp_path: Path) -> None:
    payload = build_dataset_manifest(
        env_id="mujoco/HalfCheetah-v5",
        collector_kind="mujoco_collector",
        collector_policy="policy_checkpoint",
        seed=7,
        episodes=3,
        transitions=128,
        reward_stats={"mean": 1.0, "std": 0.5},
        source_commit="abc123",
        created_at="2026-03-20T00:00:00Z",
        preprocessing={"obs_norm": False},
        artifact_paths={"replay_buffer": "datasets/halfcheetah.npz"},
    )
    manifest_path = write_dataset_manifest(tmp_path / "dataset.manifest.json", payload)

    loaded = load_dataset_manifest(manifest_path)

    assert loaded["schema_version"] == DATASET_MANIFEST_SCHEMA_VERSION
    assert loaded["collector_policy"] == "policy_checkpoint"


def test_resolve_dataset_artifact_path_relative_to_manifest(tmp_path: Path) -> None:
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "env_id": "mujoco/HalfCheetah-v5",
        "artifact_paths": {"replay_buffer": "datasets/halfcheetah.npz"},
    }
    manifest_path = tmp_path / "bundle" / "dataset.manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")

    resolved = resolve_dataset_artifact_path(
        manifest,
        artifact_key="replay_buffer",
        manifest_path=manifest_path,
    )

    assert resolved == manifest_path.parent / "datasets" / "halfcheetah.npz"


def test_load_dataset_manifest_rejects_wrong_schema(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset.manifest.json"
    manifest_path.write_text('{"schema_version":"wrong","env_id":"x"}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version"):
        load_dataset_manifest(manifest_path)
