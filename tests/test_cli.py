"""Tests for the Typer CLI entrypoint."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

if importlib.util.find_spec("typer") is None or importlib.util.find_spec("rich") is None:
    pytest.skip("CLI optional dependencies are not installed", allow_module_level=True)

from typer.testing import CliRunner

import worldflux.cli as cli

runner = CliRunner()


def _base_context(project_name: str = "demo-project", device: str = "cpu") -> dict[str, object]:
    return {
        "project_name": project_name,
        "environment": "atari",
        "model": "dreamer:ci",
        "model_type": "dreamer",
        "obs_shape": [3, 64, 64],
        "action_dim": 6,
        "hidden_dim": 32,
        "device": device,
    }


def test_init_with_path_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("named-project"))
    monkeypatch.setattr(cli, "_print_logo", lambda: None)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "my-ai"])
        assert result.exit_code == 0
        assert Path("my-ai").is_dir()
        assert Path("my-ai/worldflux.toml").exists()


def test_init_without_path_uses_project_name_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("auto-dir"))
    monkeypatch.setattr(cli, "_print_logo", lambda: None)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init"])
        assert result.exit_code == 0
        assert Path("auto-dir").is_dir()
        assert Path("auto-dir/train.py").exists()


def test_init_gpu_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context(device="cuda"))
    monkeypatch.setattr(cli, "_print_logo", lambda: None)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "gpu-fallback"])
        assert result.exit_code == 0
        content = Path("gpu-fallback/worldflux.toml").read_text(encoding="utf-8")
        assert 'device = "cpu"' in content
        assert "CUDA is not available" in result.stdout


def test_init_returns_non_zero_when_target_is_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("unused"))
    monkeypatch.setattr(cli, "_print_logo", lambda: None)

    with runner.isolated_filesystem():
        Path("target").write_text("not a dir", encoding="utf-8")
        result = runner.invoke(cli.app, ["init", "target"])
        assert result.exit_code == 1
        assert "Error:" in result.stdout


def test_resolve_python_launcher_prefers_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: "/usr/bin/uv" if cmd == "uv" else None)
    assert cli._resolve_python_launcher() == "uv run python"


def test_resolve_python_launcher_falls_back_to_python3(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(cmd: str) -> str | None:
        return "/usr/bin/python3" if cmd == "python3" else None

    monkeypatch.setattr(cli.shutil, "which", _which)
    assert cli._resolve_python_launcher() == "python3"


def test_init_next_steps_uses_resolved_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("step-launcher"))
    monkeypatch.setattr(cli, "_print_logo", lambda: None)
    monkeypatch.setattr(cli, "_resolve_python_launcher", lambda: "python3")

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init"])
        assert result.exit_code == 0
        assert "python3 train.py" in result.stdout
        assert "python3 inference.py" in result.stdout
