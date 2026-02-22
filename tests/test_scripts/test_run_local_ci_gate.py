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


def test_build_fast_gate_commands_contains_quick_ci_steps() -> None:
    mod = _load_module()
    commands = mod.build_fast_gate_commands()
    names = {command.name for command in commands}

    assert "Bandit security linter" in names
    assert "API v0.2 tests" in names
    assert "Legacy bridge tests" in names
    assert "Planner boundary tests" in names
    assert "Parity suite policy check" in names
    assert "Build docs (strict)" not in names
    assert "Build package" not in names


def test_release_checklist_gate_uses_uv_run_python(monkeypatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(mod, "_is_docs_domain_reachable", lambda timeout_seconds=5.0: True)

    commands = mod.build_gate_commands()
    release_gate = next(
        command for command in commands if command.name == "Release checklist gate wiring"
    )
    assert release_gate.argv == ("uv", "run", "python", "scripts/check_release_checklist_gate.py")


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
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def _run(argv, check=False, cwd=None):  # noqa: ARG001
        calls.append((tuple(argv), cwd))
        return type("Completed", (), {"returncode": 1 if len(calls) == 1 else 0})()

    monkeypatch.setattr(mod.subprocess, "run", _run)
    commands = (
        mod.GateCommand(name="failing", argv=("python", "-c", "1")),
        mod.GateCommand(name="would-not-run", argv=("python", "-c", "2")),
    )

    result = mod.run_commands(commands, dry_run=False, keep_going=False)
    assert result == 1
    assert len(calls) == 1
    assert calls[0][1] is not None


def test_projectize_uv_run_injects_repo_root() -> None:
    mod = _load_module()
    repo_root = Path("/tmp/worldflux")
    argv = ("uv", "run", "python", "-m", "pytest")

    updated = mod._projectize_uv_command(argv, repo_root=repo_root)
    assert updated[:4] == ("uv", "run", "--project", str(repo_root))


def test_projectize_uv_sync_injects_repo_root() -> None:
    mod = _load_module()
    repo_root = Path("/tmp/worldflux")
    argv = ("uv", "sync", "--extra", "dev")

    updated = mod._projectize_uv_command(argv, repo_root=repo_root)
    assert updated[:4] == ("uv", "sync", "--project", str(repo_root))


def test_projectize_uv_pip_is_left_unchanged() -> None:
    mod = _load_module()
    repo_root = Path("/tmp/worldflux")
    argv = ("uv", "pip", "install", "-e", ".")

    updated = mod._projectize_uv_command(argv, repo_root=repo_root)
    assert updated == argv


def test_twine_command_uses_uv_managed_twine() -> None:
    mod = _load_module()
    command = mod._twine_check_command()

    assert command.argv[:5] == ("uv", "run", "--with", "twine", "python")
    assert "twine" in command.argv[-1]
