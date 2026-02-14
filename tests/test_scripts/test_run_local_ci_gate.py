"""Tests for local CI gate runner script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_local_ci_gate.py"
    spec = importlib.util.spec_from_file_location("run_local_ci_gate", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_local_ci_gate module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_gate_commands_contains_core_ci_steps(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_is_docs_domain_reachable", lambda timeout_seconds=5.0: True)

    commands = mod.build_gate_commands()
    names = {command.name for command in commands}

    assert "Ruff lint" in names
    assert "Ruff format check" in names
    assert "Mypy" in names
    assert "Pytest (3.11)" in names
    assert "Coverage report" in names
    assert "Critical coverage thresholds" in names
    assert "Bandit security linter" in names
    assert "Build docs (strict)" in names
    assert "Check markdown links" in names
    assert "Build package" in names
    assert "Check built artifacts" in names


def test_lychee_command_excludes_worldflux_domain_when_unreachable(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_is_docs_domain_reachable", lambda timeout_seconds=5.0: False)

    command = mod._lychee_command()
    assert "--exclude" in command.argv
    assert r"worldflux\.ai" in command.argv


def test_lychee_command_does_not_exclude_when_reachable(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_is_docs_domain_reachable", lambda timeout_seconds=5.0: True)

    command = mod._lychee_command()
    assert "--exclude" not in command.argv


def test_run_commands_stops_on_first_failure(monkeypatch) -> None:
    mod = _load_module()
    calls: list[tuple[str, ...]] = []

    def _run(argv, check=False):  # noqa: ARG001
        calls.append(tuple(argv))
        return type("Completed", (), {"returncode": 1 if len(calls) == 1 else 0})()

    monkeypatch.setattr(mod.subprocess, "run", _run)
    commands = (
        mod.GateCommand(name="failing", argv=("python", "-c", "1")),
        mod.GateCommand(name="would-not-run", argv=("python", "-c", "2")),
    )

    result = mod.run_commands(commands, dry_run=False, keep_going=False)
    assert result == 1
    assert len(calls) == 1


def test_twine_command_uses_uv_managed_twine() -> None:
    mod = _load_module()
    command = mod._twine_check_command()

    assert command.argv[:5] == ("uv", "run", "--with", "twine", "python")
    assert "twine" in command.argv[-1]
