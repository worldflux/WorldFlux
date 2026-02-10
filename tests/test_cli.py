"""Tests for the Typer CLI entrypoint."""

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if importlib.util.find_spec("typer") is None or importlib.util.find_spec("rich") is None:
    pytest.skip("CLI dependencies are not installed", allow_module_level=True)

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


def _patch_init_noninteractive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_print_logo", lambda: None)
    monkeypatch.setattr(cli, "_confirm_generation", lambda: True)
    monkeypatch.setattr(cli, "_handle_optional_atari_dependency_install", lambda _ctx: None)


def test_init_with_path_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("named-project"))
    _patch_init_noninteractive(monkeypatch)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "my-ai"])
        assert result.exit_code == 0
        assert Path("my-ai").is_dir()
        assert Path("my-ai/worldflux.toml").exists()


def test_init_without_path_uses_project_name_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("auto-dir"))
    _patch_init_noninteractive(monkeypatch)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init"])
        assert result.exit_code == 0
        assert Path("auto-dir").is_dir()
        assert Path("auto-dir/train.py").exists()


def test_init_shows_guided_intro_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("intro-project"))
    monkeypatch.setattr(cli, "_confirm_generation", lambda: True)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "intro-project"])
        assert result.exit_code == 0
        assert "Create a ready-to-run WorldFlux project" in result.stdout
        assert "Configuration Summary" in result.stdout


def test_init_gpu_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context(device="cuda"))
    _patch_init_noninteractive(monkeypatch)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "gpu-fallback"])
        assert result.exit_code == 0
        content = Path("gpu-fallback/worldflux.toml").read_text(encoding="utf-8")
        assert 'device = "cpu"' in content
        assert "CUDA is not available" in result.stdout


def test_init_returns_non_zero_when_target_is_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("unused"))
    _patch_init_noninteractive(monkeypatch)

    with runner.isolated_filesystem():
        Path("target").write_text("not a dir", encoding="utf-8")
        result = runner.invoke(cli.app, ["init", "target"])
        assert result.exit_code == 1
        assert "Error:" in result.stdout


def test_resolve_python_launcher_prefers_current_uv_tool_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli.sys, "executable", "/home/user/.local/share/uv/tools/worldflux/bin/python"
    )
    monkeypatch.setattr(cli.shutil, "which", lambda _cmd: None)
    assert cli._resolve_python_launcher() == "/home/user/.local/share/uv/tools/worldflux/bin/python"


def test_resolve_python_launcher_falls_back_to_python3(monkeypatch: pytest.MonkeyPatch) -> None:
    def _which(cmd: str) -> str | None:
        return "/usr/bin/python3" if cmd == "python3" else None

    monkeypatch.setattr(cli.shutil, "which", _which)
    assert cli._resolve_python_launcher() == "python3"


def test_resolve_python_launcher_falls_back_to_uv_when_no_python_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.sys, "executable", "/opt/custom/python")
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: "/usr/bin/uv" if cmd == "uv" else None)
    assert cli._resolve_python_launcher() == "uv run python"


def test_init_next_steps_uses_resolved_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("step-launcher"))
    _patch_init_noninteractive(monkeypatch)
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
    _patch_init_noninteractive(monkeypatch)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "cancelled"])
        assert result.exit_code == 130
        assert "Initialization cancelled" in result.stdout


def test_init_decline_confirmation_returns_130_and_generates_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("declined"))
    monkeypatch.setattr(cli, "_print_logo", lambda: None)
    monkeypatch.setattr(cli, "_confirm_generation", lambda: False)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "declined"])
        assert result.exit_code == 130
        assert not Path("declined").exists()
        assert "No files were generated" in result.stdout


def test_init_force_overwrites_non_empty_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("forced-project"))
    _patch_init_noninteractive(monkeypatch)

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
    _patch_init_noninteractive(monkeypatch)

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
    monkeypatch.setattr(cli.sys, "executable", "/opt/custom/python")
    monkeypatch.setattr(cli.shutil, "which", lambda _cmd: None)
    assert cli._resolve_python_launcher() == "/opt/custom/python"


def test_missing_atari_dependency_packages_detects_expected_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _find_spec(name: str):
        if name == "gymnasium":
            return object()
        if name == "ale_py":
            return None
        return object()

    monkeypatch.setattr(cli.importlib.util, "find_spec", _find_spec)
    assert cli._missing_atari_dependency_packages() == ["ale-py"]


def test_handle_optional_atari_dependency_install_skips_non_atari(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: dict[str, bool] = {"missing_checked": False}

    def _missing() -> list[str]:
        called["missing_checked"] = True
        return ["gymnasium", "ale-py"]

    monkeypatch.setattr(cli, "_missing_atari_dependency_packages", _missing)
    cli._handle_optional_atari_dependency_install({"environment": "mujoco"})
    assert called["missing_checked"] is False


def test_handle_optional_atari_dependency_install_non_interactive_prints_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[str] = []
    monkeypatch.setattr(cli, "_missing_atari_dependency_packages", lambda: ["ale-py"])
    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: False)
    monkeypatch.setattr(cli.console, "print", lambda message="": printed.append(str(message)))
    monkeypatch.setattr(
        cli,
        "_confirm_optional_dependency_install",
        lambda _packages: pytest.fail("should not prompt in non-interactive mode"),
    )
    monkeypatch.setattr(
        cli, "_install_packages_with_pip", lambda _packages: pytest.fail("should not install")
    )

    cli._handle_optional_atari_dependency_install({"environment": "atari"})
    assert any("Install later with:" in line for line in printed)


def test_handle_optional_atari_dependency_install_interactive_and_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[list[str]] = []
    printed: list[str] = []
    monkeypatch.setattr(cli, "_missing_atari_dependency_packages", lambda: ["gymnasium", "ale-py"])
    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: True)
    monkeypatch.setattr(cli, "_confirm_optional_dependency_install", lambda _packages: True)
    monkeypatch.setattr(
        cli,
        "_install_packages_with_pip",
        lambda packages: installed.append(packages) or True,
    )
    monkeypatch.setattr(cli.console, "print", lambda message="": printed.append(str(message)))

    cli._handle_optional_atari_dependency_install({"environment": "atari"})

    assert installed == [["gymnasium", "ale-py"]]
    assert any("installed successfully" in line for line in printed)


def test_install_packages_with_pip_falls_back_to_uv_when_pip_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def _run(cmd, **_kwargs):
        calls.append(list(cmd))
        if cmd[:3] == [cli.sys.executable, "-m", "pip"]:
            return SimpleNamespace(returncode=1)
        if cmd[:4] == ["uv", "pip", "install", "--python"]:
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(cli.subprocess, "run", _run)
    monkeypatch.setattr(cli.shutil, "which", lambda cmd: "/usr/bin/uv" if cmd == "uv" else None)
    monkeypatch.setattr(cli.console, "print", lambda _message="": None)

    assert cli._install_packages_with_pip(["gymnasium", "ale-py"]) is True
    assert calls[0][:3] == [cli.sys.executable, "-m", "pip"]
    assert calls[1][:4] == ["uv", "pip", "install", "--python"]
    assert "--break-system-packages" in calls[1]


def test_install_packages_with_pip_returns_false_when_no_uv_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli.subprocess, "run", lambda _cmd, **_kwargs: SimpleNamespace(returncode=1)
    )
    monkeypatch.setattr(cli.shutil, "which", lambda _cmd: None)
    monkeypatch.setattr(cli.console, "print", lambda _message="": None)

    assert cli._install_packages_with_pip(["gymnasium", "ale-py"]) is False


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
    assert any("Recommended model" in message for message in printed)


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
    assert any("Choose your environment type" in message for message in printed)
    assert any("Invalid observation shape" in message for message in printed)
    assert any("Invalid action dim" in message for message in printed)
    assert any("CUDA is not available" in message for message in printed)
    assert any("Recommended model" in message for message in printed)


def test_confirm_generation_uses_inquirer_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_inquirer = SimpleNamespace(
        confirm=lambda **_kwargs: SimpleNamespace(execute=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "InquirerPy", SimpleNamespace(inquirer=fake_inquirer))
    assert cli._confirm_generation() is False


def test_confirm_generation_falls_back_to_rich_when_inquirer_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name == "InquirerPy":
            raise ModuleNotFoundError(name="InquirerPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    monkeypatch.setattr(cli.Confirm, "ask", staticmethod(lambda *_args, **_kwargs: True))
    assert cli._confirm_generation() is True


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
    assert len(printed) == 4
    assert "WorldFlux CLI" in printed[1]
    assert "Panel" in printed[2]


def test_cli_module_does_not_require_optional_cli_extra_hint() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert ".[cli]" not in source
