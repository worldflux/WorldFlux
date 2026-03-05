"""Tests for the release dry-run helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_release_dry_run.py"
    spec = importlib.util.spec_from_file_location("run_release_dry_run", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_release_dry_run module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_release_commands_full_contains_docs_audit_fixture_regen_and_build() -> None:
    mod = _load_module()
    commands = mod.build_release_commands(tag="v0.1.0", profile="full")
    names = {command.name for command in commands}

    assert "Docs dependency audit" in names
    assert "Regenerate release parity fixtures" in names
    assert "Build docs site" in names
    assert "Build package" in names
    assert "Check built artifacts" in names


def test_build_release_commands_verify_skips_package_build() -> None:
    mod = _load_module()
    commands = mod.build_release_commands(tag="v0.1.0", profile="verify")
    names = {command.name for command in commands}

    assert "Docs dependency audit" in names
    assert "Regenerate release parity fixtures" in names
    assert "Build docs site" in names
    assert "Build package" not in names
    assert "Check built artifacts" not in names


def test_current_tag_uses_pyproject_version(tmp_path: Path) -> None:
    mod = _load_module()
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
version = "1.2.3"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    assert mod._current_tag(tmp_path) == "v1.2.3"
