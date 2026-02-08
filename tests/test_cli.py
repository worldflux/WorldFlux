"""Tests for the Typer CLI entrypoint."""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_parse_obs_shape_accepts_comma_separated_dims() -> None:
    assert cli._parse_obs_shape("3,64,64") == [3, 64, 64]
    assert cli._parse_obs_shape(" 39 ") == [39]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        " , ",
        "3,-1,64",
        "a,b",
    ],
)
def test_parse_obs_shape_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        cli._parse_obs_shape(raw)


@pytest.mark.parametrize("raw", ["0", "-2", "abc"])
def test_parse_action_dim_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(ValueError):
        cli._parse_action_dim(raw)


def test_parse_action_dim_accepts_positive_integer() -> None:
    assert cli._parse_action_dim("6") == 6


def test_resolve_model_rules() -> None:
    assert cli._resolve_model("atari", [3, 64, 64]) == ("dreamer:ci", "dreamer")
    assert cli._resolve_model("mujoco", [39]) == ("tdmpc2:ci", "tdmpc2")
    assert cli._resolve_model("custom", [3, 64, 64]) == ("dreamer:ci", "dreamer")
    assert cli._resolve_model("custom", [39]) == ("tdmpc2:ci", "tdmpc2")


def test_init_keyboard_interrupt_returns_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def _interrupt() -> dict[str, object]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_prompt_user_configuration", _interrupt)
    monkeypatch.setattr(cli, "_print_logo", lambda: None)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "cancelled"])
        assert result.exit_code == 130
        assert "Initialization cancelled" in result.stdout


def test_init_force_overwrites_non_empty_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("forced-project"))
    monkeypatch.setattr(cli, "_print_logo", lambda: None)

    with runner.isolated_filesystem():
        target = Path("forced")
        target.mkdir(parents=True)
        (target / "keep.txt").write_text("existing", encoding="utf-8")

        first = runner.invoke(cli.app, ["init", "forced"])
        assert first.exit_code == 1

        second = runner.invoke(cli.app, ["init", "forced", "--force"])
        assert second.exit_code == 0
        assert (target / "worldflux.toml").exists()


def test_init_value_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("bad-context"))
    monkeypatch.setattr(cli, "_print_logo", lambda: None)

    def _raise(*args, **kwargs):
        raise ValueError("invalid config")

    monkeypatch.setattr(cli, "generate_project", _raise)
    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "target"])
        assert result.exit_code == 1
        assert "invalid config" in result.stdout


def test_resolve_python_launcher_defaults_to_python_when_no_launcher_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.shutil, "which", lambda _cmd: None)
    assert cli._resolve_python_launcher() == "python"


def test_prompt_with_inquirer_returns_none_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name == "InquirerPy":
            raise ModuleNotFoundError(name="InquirerPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert cli._prompt_with_inquirer() is None


def test_prompt_with_inquirer_retries_invalid_values_and_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_answers = iter(
        [
            "advanced-project",  # project name
            "invalid-shape",  # observation shape (invalid)
            "3,64,64",  # observation shape (valid)
            "bad-action",  # action dim (invalid)
            "6",  # action dim (valid)
        ]
    )
    printed: list[str] = []

    def _text(*_args, **_kwargs):
        return SimpleNamespace(execute=lambda: next(text_answers))

    fake_inquirer = SimpleNamespace(
        text=_text,
        select=lambda **_kwargs: SimpleNamespace(execute=lambda: "custom"),
        confirm=lambda **_kwargs: SimpleNamespace(execute=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "InquirerPy", SimpleNamespace(inquirer=fake_inquirer))
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.console, "print", lambda message="": printed.append(str(message)))

    config = cli._prompt_with_inquirer()

    assert config is not None
    assert config["project_name"] == "advanced-project"
    assert config["obs_shape"] == [3, 64, 64]
    assert config["action_dim"] == 6
    assert config["model"] == "dreamer:ci"
    assert config["device"] == "cpu"
    assert any("Invalid observation shape" in message for message in printed)
    assert any("Invalid action dim" in message for message in printed)
    assert any("CUDA is not available" in message for message in printed)


def test_prompt_with_rich_retries_invalid_values_and_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_answers = iter(
        [
            "",  # project name (invalid)
            "rich-project",  # project name (valid)
            "custom",  # environment
            "oops",  # observation shape (invalid)
            "39",  # observation shape (valid)
            "nan",  # action dim (invalid)
            "6",  # action dim (valid)
        ]
    )
    printed: list[str] = []
    monkeypatch.setattr(
        cli.Prompt, "ask", staticmethod(lambda *_args, **_kwargs: next(prompt_answers))
    )
    monkeypatch.setattr(cli.Confirm, "ask", staticmethod(lambda *_args, **_kwargs: True))
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.console, "print", lambda message="": printed.append(str(message)))

    config = cli._prompt_with_rich()

    assert config["project_name"] == "rich-project"
    assert config["obs_shape"] == [39]
    assert config["action_dim"] == 6
    assert config["model"] == "tdmpc2:ci"
    assert config["device"] == "cpu"
    assert any("Project name cannot be empty" in message for message in printed)
    assert any("Invalid observation shape" in message for message in printed)
    assert any("Invalid action dim" in message for message in printed)
    assert any("CUDA is not available" in message for message in printed)


def test_prompt_user_configuration_prefers_inquirer_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_context()
    monkeypatch.setattr(cli, "_prompt_with_inquirer", lambda: config)
    monkeypatch.setattr(
        cli, "_prompt_with_rich", lambda: pytest.fail("rich fallback should not be called")
    )
    assert cli._prompt_user_configuration() == config


def test_prompt_user_configuration_falls_back_to_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_context("rich-fallback")
    monkeypatch.setattr(cli, "_prompt_with_inquirer", lambda: None)
    monkeypatch.setattr(cli, "_prompt_with_rich", lambda: config)
    assert cli._prompt_user_configuration() == config


def test_print_logo_writes_banner_and_header(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []
    monkeypatch.setattr(cli.console, "print", lambda message="": printed.append(str(message)))
    cli._print_logo()
    assert len(printed) == 3
    assert "WorldFlux CLI" in printed[1]
