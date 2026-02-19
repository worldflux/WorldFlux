"""Tests for the Typer CLI entrypoint."""

from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
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
        "training_total_steps": 100000,
        "training_batch_size": 16,
        "hidden_dim": 32,
        "device": device,
    }


def _patch_init_noninteractive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_print_logo", lambda: None)
    monkeypatch.setattr(cli, "_confirm_generation", lambda: True)
    monkeypatch.setattr(cli, "_handle_optional_atari_dependency_install", lambda _ctx: None)
    monkeypatch.setattr(
        cli,
        "ensure_init_dependencies",
        lambda _ctx, **_kwargs: SimpleNamespace(
            success=True,
            skipped=True,
            launcher=None,
            message="bootstrap skipped",
            retry_commands=(),
            diagnostics=(),
        ),
    )


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


def test_init_exits_when_dependency_bootstrap_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("bootstrap-fail"))
    monkeypatch.setattr(cli, "_print_logo", lambda: None)
    monkeypatch.setattr(
        cli,
        "ensure_init_dependencies",
        lambda _ctx, **_kwargs: SimpleNamespace(
            success=False,
            skipped=False,
            launcher=None,
            message="deps failed",
            retry_commands=("python -m venv ~/.worldflux/bootstrap/py311",),
            diagnostics=("pip install failed",),
        ),
    )

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "bootstrap-fail"])
        assert result.exit_code == 1
        assert "Dependency bootstrap failed" in result.stdout
        assert "pip install failed" in result.stdout
        assert not Path("bootstrap-fail").exists()


def test_init_uses_bootstrap_launcher_in_next_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli, "_prompt_user_configuration", lambda: _base_context("bootstrap-launcher")
    )
    monkeypatch.setattr(cli, "_print_logo", lambda: None)
    monkeypatch.setattr(cli, "_confirm_generation", lambda: True)
    monkeypatch.setattr(
        cli,
        "ensure_init_dependencies",
        lambda _ctx, **_kwargs: SimpleNamespace(
            success=True,
            skipped=False,
            launcher="/tmp/bootstrap/bin/python",
            message="deps ready",
            retry_commands=(),
            diagnostics=(),
        ),
    )

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init"])
        assert result.exit_code == 0
        assert "/tmp/bootstrap/bin/python train.py" in result.stdout
        assert "/tmp/bootstrap/bin/python inference.py" in result.stdout


def test_init_shows_guided_intro_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context("intro-project"))
    monkeypatch.setattr(cli, "_confirm_generation", lambda: True)
    monkeypatch.setattr(
        cli,
        "ensure_init_dependencies",
        lambda _ctx, **_kwargs: SimpleNamespace(
            success=True,
            skipped=True,
            launcher=None,
            message="bootstrap skipped",
            retry_commands=(),
            diagnostics=(),
        ),
    )

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init", "intro-project"])
        assert result.exit_code == 0
        assert "Create a ready-to-run WorldFlux project" in result.stdout
        assert "Configuration Summary" in result.stdout
        assert cli.OBS_ACTION_GUIDE_URL in result.stdout
        assert "Model fit:" in result.stdout


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


def test_init_shows_bootstrap_progress_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli, "_prompt_user_configuration", lambda: _base_context("progress-project")
    )
    monkeypatch.setattr(cli, "_print_logo", lambda: None)
    monkeypatch.setattr(cli, "_confirm_generation", lambda: True)

    def _ensure(_ctx, **kwargs):
        progress_callback = kwargs.get("progress_callback")
        assert callable(progress_callback)
        progress_callback("Installing dependencies...")
        return SimpleNamespace(
            success=True,
            skipped=False,
            launcher=None,
            message="deps ready",
            retry_commands=(),
            diagnostics=(),
        )

    monkeypatch.setattr(cli, "ensure_init_dependencies", _ensure)

    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["init"])
        assert result.exit_code == 0
        assert "Preparing bootstrap dependencies before project generation" in result.stdout
        assert "Installing dependencies..." in result.stdout


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


@pytest.mark.parametrize(
    ("raw", "field_name"),
    [
        ("0", "Total training steps"),
        ("-1", "Batch size"),
        ("abc", "Batch size"),
    ],
)
def test_parse_positive_int_rejects_invalid_values(raw: str, field_name: str) -> None:
    with pytest.raises(ValueError):
        cli._parse_positive_int(raw, field_name=field_name)


def test_parse_positive_int_accepts_positive_integer() -> None:
    assert cli._parse_positive_int("64", field_name="Batch size") == 64


def test_resolve_model_rules() -> None:
    assert cli._resolve_model("atari", [3, 64, 64]) == ("dreamer:ci", "dreamer")
    assert cli._resolve_model("mujoco", [39]) == ("tdmpc2:ci", "tdmpc2")
    assert cli._resolve_model("custom", [3, 64, 64]) == ("dreamer:ci", "dreamer")
    assert cli._resolve_model("custom", [39]) == ("tdmpc2:ci", "tdmpc2")


def test_model_type_from_model_id() -> None:
    assert cli._model_type_from_model_id("dreamer:ci") == "dreamer"
    assert cli._model_type_from_model_id("tdmpc2:ci") == "tdmpc2"
    with pytest.raises(ValueError):
        cli._model_type_from_model_id("unknown:ci")


def test_select_model_with_inquirer_uses_recommended_default() -> None:
    captured: dict[str, object] = {}

    def _select(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(execute=lambda: "dreamer:ci")

    fake_inquirer = SimpleNamespace(select=_select)
    selected = cli._select_model_with_inquirer(fake_inquirer, "dreamer:ci")

    assert selected == "dreamer:ci"
    assert captured["default"] == "dreamer:ci"
    assert str(captured["message"]).startswith("Choose model")
    choices = captured["choices"]
    assert isinstance(choices, list)
    assert choices[0]["value"] == "dreamer:ci"
    assert choices[1]["value"] == "tdmpc2:ci"


def test_select_model_with_rich_uses_recommended_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _ask(*_args, **kwargs):
        captured.update(kwargs)
        return "tdmpc2:ci"

    monkeypatch.setattr(cli.Prompt, "ask", staticmethod(_ask))
    selected = cli._select_model_with_rich("tdmpc2:ci")

    assert selected == "tdmpc2:ci"
    assert captured["default"] == "tdmpc2:ci"
    choices = captured["choices"]
    assert isinstance(choices, list)
    assert choices[0] == "tdmpc2:ci"
    assert choices[1] == "dreamer:ci"


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
    monkeypatch.setattr(
        cli,
        "ensure_init_dependencies",
        lambda _ctx, **_kwargs: SimpleNamespace(
            success=True,
            skipped=True,
            launcher=None,
            message="bootstrap skipped",
            retry_commands=(),
            diagnostics=(),
        ),
    )

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
            "0",  # total steps (invalid)
            "120000",  # total steps (valid)
            "bad-batch",  # batch size (invalid)
            "32",  # batch size (valid)
        ]
    )
    select_answers = iter(
        [
            "custom",  # environment
            "dreamer:ci",  # model selection
        ]
    )
    printed: list[str] = []
    model_choices_called: list[str] = []

    def _text(*_args, **_kwargs):
        return SimpleNamespace(execute=lambda: next(text_answers))

    fake_inquirer = SimpleNamespace(
        text=_text,
        select=lambda **_kwargs: SimpleNamespace(execute=lambda: next(select_answers)),
        confirm=lambda **_kwargs: SimpleNamespace(execute=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "InquirerPy", SimpleNamespace(inquirer=fake_inquirer))
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.console, "print", lambda message="": printed.append(str(message)))
    monkeypatch.setattr(
        cli,
        "_print_model_choices",
        lambda recommended_model: model_choices_called.append(recommended_model),
    )

    config = cli._prompt_with_inquirer()

    assert config is not None
    assert config["project_name"] == "advanced-project"
    assert config["obs_shape"] == [3, 64, 64]
    assert config["action_dim"] == 6
    assert config["training_total_steps"] == 120000
    assert config["training_batch_size"] == 32
    assert config["model"] == "dreamer:ci"
    assert config["device"] == "cpu"
    assert any("Invalid observation shape" in message for message in printed)
    assert any("Invalid action dim" in message for message in printed)
    assert any("Invalid total steps" in message for message in printed)
    assert any("Invalid batch size" in message for message in printed)
    assert any("CUDA is not available" in message for message in printed)
    assert any("Recommended model" in message for message in printed)
    assert model_choices_called == ["dreamer:ci"]


def test_prompt_with_inquirer_allows_alternative_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_answers = iter(
        [
            "alt-project",
            "3,64,64",
            "6",
            "100000",
            "16",
        ]
    )
    select_answers = iter(
        [
            "atari",
            "tdmpc2:ci",
        ]
    )
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)

    def _text(*_args, **_kwargs):
        return SimpleNamespace(execute=lambda: next(text_answers))

    fake_inquirer = SimpleNamespace(
        text=_text,
        select=lambda **_kwargs: SimpleNamespace(execute=lambda: next(select_answers)),
        confirm=lambda **_kwargs: SimpleNamespace(execute=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "InquirerPy", SimpleNamespace(inquirer=fake_inquirer))

    config = cli._prompt_with_inquirer()

    assert config is not None
    assert config["model"] == "tdmpc2:ci"
    assert config["model_type"] == "tdmpc2"


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
            "tdmpc2:ci",  # selected model
            "0",  # total steps (invalid)
            "80000",  # total steps (valid)
            "oops",  # batch size (invalid)
            "24",  # batch size (valid)
        ]
    )
    printed: list[str] = []
    model_choices_called: list[str] = []
    monkeypatch.setattr(
        cli.Prompt, "ask", staticmethod(lambda *_args, **_kwargs: next(prompt_answers))
    )
    monkeypatch.setattr(cli.Confirm, "ask", staticmethod(lambda *_args, **_kwargs: True))
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.console, "print", lambda message="": printed.append(str(message)))
    monkeypatch.setattr(
        cli,
        "_print_model_choices",
        lambda recommended_model: model_choices_called.append(recommended_model),
    )

    config = cli._prompt_with_rich()

    assert config["project_name"] == "rich-project"
    assert config["obs_shape"] == [39]
    assert config["action_dim"] == 6
    assert config["training_total_steps"] == 80000
    assert config["training_batch_size"] == 24
    assert config["model"] == "tdmpc2:ci"
    assert config["device"] == "cpu"
    assert any("Project name cannot be empty" in message for message in printed)
    assert any("Choose your environment type" in message for message in printed)
    assert any("Invalid observation shape" in message for message in printed)
    assert any("Invalid action dim" in message for message in printed)
    assert any("Invalid total steps" in message for message in printed)
    assert any("Invalid batch size" in message for message in printed)
    assert any("CUDA is not available" in message for message in printed)
    assert any("Recommended model" in message for message in printed)
    assert model_choices_called == ["tdmpc2:ci"]


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


def test_parity_run_enforce_exits_non_zero_on_failed_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "run_suite",
        lambda *_args, **_kwargs: {
            "suite": {"suite_id": "suite_a", "family": "dreamerv3"},
            "stats": {
                "pass_non_inferiority": False,
                "sample_size": 20,
                "mean_drop_ratio": 0.06,
                "ci_upper_ratio": 0.07,
                "margin_ratio": 0.05,
            },
        },
    )

    result = runner.invoke(cli.app, ["parity", "run", "suite.yaml", "--enforce"])
    assert result.exit_code == 1
    assert "Parity Run (Legacy)" in result.stdout
    assert "legacy non-inferiority harness" in result.stdout


def test_parity_run_reports_failure_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args, **_kwargs):
        raise cli.ParityError("boom")

    monkeypatch.setattr(cli, "run_suite", _raise)
    result = runner.invoke(cli.app, ["parity", "run", "suite.yaml"])
    assert result.exit_code == 1
    assert "Parity run failed" in result.stdout


def test_parity_aggregate_fails_when_no_artifacts_found() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(cli.app, ["parity", "aggregate", "--runs-glob", "missing/*.json"])
    assert result.exit_code == 1
    assert "No run artifacts found to aggregate" in result.stdout


def test_parity_aggregate_enforce_exits_on_failed_aggregate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "aggregate_runs",
        lambda *_args, **_kwargs: {
            "all_suites_pass": False,
            "run_count": 2,
            "suite_pass_count": 1,
            "suite_fail_count": 1,
        },
    )

    result = runner.invoke(
        cli.app,
        ["parity", "aggregate", "--run", "run_a.json", "--run", "run_b.json", "--enforce"],
    )
    assert result.exit_code == 1
    assert "Parity Aggregate (Legacy)" in result.stdout
    assert "legacy non-inferiority harness" in result.stdout


def test_parity_report_writes_output_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    with runner.isolated_filesystem():
        aggregate = Path("aggregate.json")
        output = Path("report.md")
        aggregate.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(cli, "render_markdown_report", lambda _payload: "# Report\n")

        result = runner.invoke(
            cli.app,
            ["parity", "report", "--aggregate", str(aggregate), "--output", str(output)],
        )

        assert result.exit_code == 0
        assert output.exists()
        assert output.read_text(encoding="utf-8") == "# Report\n"


def test_parity_proof_run_invokes_matrix_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, list[str]]] = []

    def _run(script_name: str, args: list[str]) -> str:
        calls.append((script_name, list(args)))
        return '{"run_id":"proof_01"}'

    monkeypatch.setattr(cli, "_run_parity_proof_script", _run)

    result = runner.invoke(
        cli.app,
        [
            "parity",
            "proof-run",
            str(tmp_path / "manifest.yaml"),
            "--run-id",
            "proof_01",
            "--seed-list",
            "0,1",
            "--task-filter",
            "atari*",
        ],
    )

    assert result.exit_code == 0
    assert calls
    assert calls[0][0] == "run_parity_matrix.py"
    assert "--manifest" in calls[0][1]
    assert "--run-id" in calls[0][1]
    assert "--seed-list" in calls[0][1]
    assert "--task-filter" in calls[0][1]
    assert "Parity Proof Run" in result.stdout


def test_parity_proof_report_runs_completeness_stats_and_markdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run_01"
    run_root.mkdir(parents=True)
    runs = run_root / "parity_runs.jsonl"
    runs.write_text("", encoding="utf-8")
    (run_root / "seed_plan.json").write_text('{"seed_values":[0]}', encoding="utf-8")
    (run_root / "run_context.json").write_text('{"seeds":[0]}', encoding="utf-8")
    equivalence = run_root / "equivalence_report.json"
    markdown = run_root / "equivalence_report.md"

    calls: list[str] = []

    def _run(script_name: str, args: list[str]) -> str:
        calls.append(script_name)
        if script_name == "stats_equivalence.py":
            output = Path(args[args.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {
                        "global": {
                            "parity_pass_final": True,
                            "validity_pass": True,
                            "missing_pairs": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            validity_path = Path(args[args.index("--validity-report") + 1])
            validity_path.write_text("{}", encoding="utf-8")
        if script_name == "report_markdown.py":
            output = Path(args[args.index("--output") + 1])
            output.write_text("# Proof\n", encoding="utf-8")
        if script_name == "validate_matrix_completeness.py":
            output = Path(args[args.index("--output") + 1])
            output.write_text('{"missing_pairs":0,"pass":true}', encoding="utf-8")
        return ""

    monkeypatch.setattr(cli, "_run_parity_proof_script", _run)

    result = runner.invoke(
        cli.app,
        [
            "parity",
            "proof-report",
            str(tmp_path / "manifest.yaml"),
            "--runs",
            str(runs),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        "validate_matrix_completeness.py",
        "stats_equivalence.py",
        "report_markdown.py",
    ]
    assert equivalence.exists()
    assert markdown.exists()
    assert "Parity Proof Report" in result.stdout


def test_parity_report_rejects_non_object_payload() -> None:
    with runner.isolated_filesystem():
        aggregate = Path("aggregate.json")
        aggregate.write_text("[]", encoding="utf-8")
        result = runner.invoke(cli.app, ["parity", "report", "--aggregate", str(aggregate)])
    assert result.exit_code == 1
    assert "Parity report failed" in result.stdout


def test_resolve_campaign_seeds_prefers_cli_values() -> None:
    assert cli._resolve_campaign_seeds((5, 6), "2,1,2") == (1, 2)


def test_resolve_campaign_seeds_falls_back_to_spec_default() -> None:
    assert cli._resolve_campaign_seeds((4, 9), None) == (4, 9)


def test_resolve_campaign_seeds_raises_without_any_values() -> None:
    with pytest.raises(cli.ParityError, match="No seeds provided"):
        cli._resolve_campaign_seeds((), None)


def test_parity_campaign_run_builds_run_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}
    spec = SimpleNamespace(default_seeds=(7, 8))

    def _run_campaign(run_spec, run_options):
        captured["spec"] = run_spec
        captured["options"] = run_options
        return {"suite_id": "campaign_suite"}

    monkeypatch.setattr(cli, "load_campaign_spec", lambda _campaign: spec)
    monkeypatch.setattr(cli, "run_campaign", _run_campaign)
    monkeypatch.setattr(cli, "render_campaign_summary", lambda _summary: "campaign summary")

    output = tmp_path / "worldflux.json"
    oracle_output = tmp_path / "oracle.json"
    workdir = tmp_path / "workdir"
    pair_root = tmp_path / "pairs"

    result = runner.invoke(
        cli.app,
        [
            "parity",
            "campaign",
            "run",
            str(tmp_path / "campaign.yaml"),
            "--mode",
            "both",
            "--seeds",
            "3,1,3",
            "--device",
            "cuda",
            "--output",
            str(output),
            "--oracle-output",
            str(oracle_output),
            "--workdir",
            str(workdir),
            "--pair-output-root",
            str(pair_root),
            "--no-resume",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    options = captured["options"]
    assert isinstance(options, cli.CampaignRunOptions)
    assert options.mode == "both"
    assert options.seeds == (1, 3)
    assert options.device == "cuda"
    assert options.output == output.resolve()
    assert options.oracle_output == oracle_output.resolve()
    assert options.resume is False
    assert options.dry_run is True
    assert options.workdir == workdir.resolve()
    assert options.pair_output_root == pair_root.resolve()


def test_parity_campaign_run_handles_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cli, "load_campaign_spec", lambda _campaign: SimpleNamespace(default_seeds=(1,))
    )

    def _raise(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["failing-command"])

    monkeypatch.setattr(cli, "run_campaign", _raise)
    result = runner.invoke(cli.app, ["parity", "campaign", "run", str(tmp_path / "campaign.yaml")])
    assert result.exit_code == 1
    assert "Parity campaign failed" in result.stdout


def test_parity_campaign_resume_delegates_to_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def _run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "parity_campaign_run", _run)
    result = runner.invoke(
        cli.app,
        [
            "parity",
            "campaign",
            "resume",
            str(tmp_path / "campaign.yaml"),
            "--mode",
            "oracle",
            "--seeds",
            "4,2",
            "--device",
            "cuda",
        ],
    )

    assert result.exit_code == 0
    assert captured["mode"] == "oracle"
    assert captured["seeds"] == "4,2"
    assert captured["device"] == "cuda"
    assert captured["resume"] is True
    assert captured["dry_run"] is False


def test_parity_campaign_export_rejects_invalid_source(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["parity", "campaign", "export", str(tmp_path / "campaign.yaml"), "--source", "invalid"],
    )
    assert result.exit_code == 1
    assert "--source must be one of" in result.stdout


def test_parity_campaign_export_invokes_exporter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    spec = SimpleNamespace(default_seeds=(9, 10))

    def _export(run_spec, *, source_name: str, seeds, output_path, resume: bool):
        captured["spec"] = run_spec
        captured["source_name"] = source_name
        captured["seeds"] = seeds
        captured["output_path"] = output_path
        captured["resume"] = resume
        return {"suite_id": "campaign_suite"}

    monkeypatch.setattr(cli, "load_campaign_spec", lambda _campaign: spec)
    monkeypatch.setattr(cli, "export_campaign_source", _export)
    monkeypatch.setattr(cli, "render_campaign_summary", lambda _summary: "campaign summary")

    output_path = tmp_path / "export.json"
    result = runner.invoke(
        cli.app,
        [
            "parity",
            "campaign",
            "export",
            str(tmp_path / "campaign.yaml"),
            "--source",
            "oracle",
            "--seeds",
            "2,2,1",
            "--output",
            str(output_path),
            "--no-resume",
        ],
    )

    assert result.exit_code == 0
    assert captured["source_name"] == "oracle"
    assert captured["seeds"] == (1, 2)
    assert captured["output_path"] == output_path.resolve()
    assert captured["resume"] is False


def test_cli_module_does_not_require_optional_cli_extra_hint() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert ".[cli]" not in source
