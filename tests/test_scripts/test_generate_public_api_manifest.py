# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for the public API stability manifest generator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    sys.modules.pop("worldflux", None)
    script_path = repo_root / "scripts" / "generate_public_api_manifest.py"
    spec = importlib.util.spec_from_file_location("generate_public_api_manifest", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load generate_public_api_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_api_manifest_contains_stability_tiers() -> None:
    mod = _load_script_module()
    manifest = mod.generate_manifest()

    assert manifest["schema_version"] == "worldflux.public_api_manifest.v1"
    assert "worldflux.create_world_model" in manifest["symbols"]
    assert manifest["symbols"]["worldflux.create_world_model"]["stability"] == "stable"


def test_generated_api_manifest_includes_module_path_and_export_source() -> None:
    mod = _load_script_module()
    manifest = mod.generate_manifest()

    entry = manifest["symbols"]["worldflux.create_world_model"]
    assert entry["module"] == "worldflux.factory"
    assert entry["exported_from"] == "worldflux"
