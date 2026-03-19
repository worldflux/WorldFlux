# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for official DreamerV3 wrapper helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "parity"
        / "wrappers"
        / "official_dreamerv3.py"
    )
    spec = importlib.util.spec_from_file_location("official_dreamerv3", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load official wrapper")
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_official_env_includes_repo_and_embodied_paths(tmp_path: Path) -> None:
    mod = _load_module()
    repo_root = tmp_path / "dreamerv3-official"
    env = mod._official_env(repo_root=repo_root)
    pythonpath = env["PYTHONPATH"].split(mod.os.pathsep)
    assert str(repo_root.resolve()) in pythonpath
    assert str((repo_root / "embodied").resolve()) in pythonpath
