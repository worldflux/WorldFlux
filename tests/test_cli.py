# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for the Typer CLI entrypoint."""

from __future__ import annotations

import builtins
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if importlib.util.find_spec("typer") is None or importlib.util.find_spec("rich") is None:
    pytest.skip("CLI dependencies are not installed", allow_module_level=True)

from typer.testing import CliRunner

import worldflux.cli as cli

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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
        assert "worldflux train" in result.stdout
        assert "worldflux verify" in result.stdout


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
        assert "Best for" in result.stdout


def test_init_gpu_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_prompt_user_configuration", lambda: _base_context(device="cuda"))
    _patch_init_noninteractive(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

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


def test_root_help_promotes_supported_verify_workflow() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    normalized = " ".join(_ANSI_ESCAPE_RE.sub("", result.stdout).split())
    assert "worldflux verify --target ./outputs --mode quick" in normalized
    assert "worldflux eval ./outputs --suite quick" not in result.stdout


def test_train_delegated_backend_uses_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    from worldflux.training import JobHandle, JobStatus

    class _FakeBackend:
        def submit(self, config: dict[str, object]) -> JobHandle:
            assert config["backend_profile"] == "official_xl"
            return JobHandle(
                job_id="train_dreamer_42",
                backend="official_dreamerv3_jax_subprocess",
                metadata={
                    "execution_result": {
                        "status": "succeeded",
                        "reason_code": "none",
                        "summary_path": "/tmp/summary.json",
                        "next_action": None,
                    }
                },
            )

        def status(self, handle: JobHandle) -> JobStatus:
            return JobStatus.COMPLETED

        def logs(self, handle: JobHandle):
            return iter(())

        def cancel(self, handle: JobHandle) -> None:
            return None

    monkeypatch.setattr(
        "worldflux.training.trainer.ExecutionDelegatingBackend",
        lambda: _FakeBackend(),
    )

    with runner.isolated_filesystem():
        Path("worldflux.toml").write_text(
            """\
project_name = "delegated-train"
environment = "atari"
model = "dreamerv3:official_xl"

[architecture]
obs_shape = [3, 64, 64]
action_dim = 6
hidden_dim = 32

[training]
total_steps = 5
batch_size = 4
sequence_length = 10
device = "cpu"
backend = "official_dreamerv3_jax_subprocess"
backend_profile = "official_xl"
output_dir = "./delegated-out"

[verify]
env = "atari/pong"
""",
            encoding="utf-8",
        )
        result = runner.invoke(cli.app, ["train"])
        assert result.exit_code == 0
        assert "Delegated Training Result" in result.stdout
        assert "official_dreamerv3_jax_subprocess" in result.stdout
        assert "train_dreamer_42" in result.stdout


def test_train_delegated_backend_defaults_profile_from_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldflux.training import JobHandle, JobStatus

    class _FakeBackend:
        def submit(self, config: dict[str, object]) -> JobHandle:
            assert config["backend_profile"] == "official_xl"
            return JobHandle(
                job_id="train_dreamer_43",
                backend="official_dreamerv3_jax_subprocess",
                metadata={
                    "execution_result": {
                        "status": "succeeded",
                        "reason_code": "none",
                        "summary_path": "/tmp/summary.json",
                        "next_action": None,
                    }
                },
            )

        def status(self, handle: JobHandle) -> JobStatus:
            return JobStatus.COMPLETED

        def logs(self, handle: JobHandle):
            return iter(())

        def cancel(self, handle: JobHandle) -> None:
            return None

    monkeypatch.setattr(
        "worldflux.training.trainer.ExecutionDelegatingBackend",
        lambda: _FakeBackend(),
    )

    with runner.isolated_filesystem():
        Path("worldflux.toml").write_text(
            """\
project_name = "delegated-train-default-profile"
environment = "atari"
model = "dreamerv3:official_xl"

[architecture]
obs_shape = [3, 64, 64]
action_dim = 6
hidden_dim = 32

[training]
total_steps = 5
batch_size = 4
sequence_length = 10
device = "cpu"
backend = "official_dreamerv3_jax_subprocess"
output_dir = "./delegated-out"

[verify]
env = "atari/pong"
""",
            encoding="utf-8",
        )
        result = runner.invoke(cli.app, ["train"])
        assert result.exit_code == 0
        assert "official_xl" in result.stdout
        assert "train_dreamer_43" in result.stdout


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
        assert "worldflux train" in result.stdout
        assert "worldflux verify --target ./outputs --mode quick" in result.stdout


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
        return 1  # first option (default)

    monkeypatch.setattr(cli.IntPrompt, "ask", staticmethod(_ask))
    selected = cli._select_model_with_rich("tdmpc2:ci")

    assert selected == "tdmpc2:ci"
    assert captured["default"] == 1
    choices = captured["choices"]
    assert isinstance(choices, list)
    assert choices[0] == "1"
    assert choices[1] == "2"


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
    monkeypatch.setattr(
        cli.console, "print", lambda message="", **_kw: printed.append(str(message))
    )
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
    monkeypatch.setattr(
        cli.console, "print", lambda message="", **_kw: printed.append(str(message))
    )

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
            "custom",  # training steps (triggers text prompt)
            "custom",  # batch size (triggers text prompt)
            "cuda",  # device (CUDA not available → falls back to CPU)
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
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        cli.console, "print", lambda message="", **_kw: printed.append(str(message))
    )
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
            "100000",
            "16",
        ]
    )
    select_answers = iter(
        [
            "atari",
            "tdmpc2:ci",
            "custom",  # training steps (triggers text prompt)
            "custom",  # batch size (triggers text prompt)
            "cuda",  # device (CUDA available)
        ]
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

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
            "oops",  # observation shape (invalid)
            "39",  # observation shape (valid)
            "nan",  # action dim (invalid)
            "6",  # action dim (valid)
            "0",  # total steps (invalid)
            "80000",  # total steps (valid)
            "oops",  # batch size (invalid)
            "24",  # batch size (valid)
        ]
    )
    # _numbered_select answers: environment, steps, batch, device
    numbered_answers = iter(
        [
            "custom",  # environment
            "custom",  # steps preset → triggers text prompt
            "custom",  # batch preset → triggers text prompt
            "cuda",  # device (CUDA unavailable → falls back to CPU)
        ]
    )
    printed: list[str] = []
    model_choices_called: list[str] = []
    monkeypatch.setattr(
        cli.Prompt, "ask", staticmethod(lambda *_args, **_kwargs: next(prompt_answers))
    )
    monkeypatch.setattr(cli.Confirm, "ask", staticmethod(lambda *_args, **_kwargs: True))
    monkeypatch.setattr(cli, "_numbered_select", lambda *_args, **_kwargs: next(numbered_answers))
    monkeypatch.setattr(cli, "_select_model_with_rich", lambda _recommended: "tdmpc2:ci")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        cli.console, "print", lambda message="", **_kw: printed.append(str(message))
    )
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


def test_confirm_generation_skips_inquirer_when_terminal_is_not_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: False)
    monkeypatch.setattr(
        cli.Confirm,
        "ask",
        staticmethod(lambda *_args, **_kwargs: True),
    )
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


def test_prompt_user_configuration_skips_inquirer_when_terminal_is_not_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _base_context("non-interactive")
    monkeypatch.setattr(cli, "_is_interactive_terminal", lambda: False)
    monkeypatch.setattr(
        cli,
        "_prompt_with_inquirer",
        lambda: pytest.fail("inquirer prompt should not run without a tty"),
    )
    monkeypatch.setattr(cli, "_prompt_with_rich", lambda: config)
    assert cli._prompt_user_configuration() == config


def test_print_logo_writes_banner_and_header(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[object] = []
    monkeypatch.setattr(cli.console, "print", lambda message="", **_kw: printed.append(message))
    cli._print_logo()
    assert len(printed) == 4
    # printed[0] is brand Panel, printed[2] is welcome Panel
    from rich.panel import Panel

    assert isinstance(printed[0], Panel)
    assert isinstance(printed[2], Panel)


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


def test_parity_proof_run_resolves_manifest_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[str]]] = []

    def _run(script_name: str, args: list[str]) -> str:
        calls.append((script_name, list(args)))
        return '{"run_id":"proof_02"}'

    monkeypatch.setattr(cli, "_run_parity_proof_script", _run)

    result = runner.invoke(
        cli.app,
        [
            "parity",
            "proof-run",
            "--family",
            "dreamer",
            "--allow-official-only",
            "--seed-list",
            "0,1,2,3,4,5,6,7,8,9",
        ],
    )

    assert result.exit_code == 0
    manifest_arg = calls[0][1][calls[0][1].index("--manifest") + 1]
    assert manifest_arg.endswith("dreamerv3_official_checkpoint_bootstrap_v1.json")


def test_resolve_proof_manifest_uses_dreamer_canonical_backend_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import worldflux.cli._parity as parity_cli

    captured: dict[str, object] = {}
    manifest_path = tmp_path / "dreamer.yaml"

    def _resolve(request, *, scripts_root: Path, allow_official_only: bool):
        captured["backend"] = request.backend
        captured["family"] = request.family
        captured["mode"] = request.mode
        captured["scripts_root"] = scripts_root
        captured["allow_official_only"] = allow_official_only
        return SimpleNamespace(manifest_path=manifest_path, early_result=None)

    monkeypatch.setattr(parity_cli, "resolve_execution_manifest", _resolve)

    resolved = parity_cli._resolve_proof_manifest(
        manifest=None,
        family="dreamer",
        backend="",
        allow_official_only=True,
        seed_list="0,1,2,3,4,5,6,7,8,9",
    )

    assert resolved == manifest_path
    assert captured["backend"] == "official_dreamerv3_jax_subprocess"
    assert captured["family"] == "dreamer"
    assert captured["mode"] == "proof_bootstrap"
    assert captured["allow_official_only"] is True


def test_resolve_proof_manifest_uses_tdmpc2_canonical_backend_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import worldflux.cli._parity as parity_cli

    captured: dict[str, object] = {}
    manifest_path = tmp_path / "tdmpc2.yaml"

    def _resolve(request, *, scripts_root: Path, allow_official_only: bool):
        captured["backend"] = request.backend
        captured["family"] = request.family
        captured["mode"] = request.mode
        captured["scripts_root"] = scripts_root
        captured["allow_official_only"] = allow_official_only
        return SimpleNamespace(manifest_path=manifest_path, early_result=None)

    monkeypatch.setattr(parity_cli, "resolve_execution_manifest", _resolve)

    resolved = parity_cli._resolve_proof_manifest(
        manifest=None,
        family="tdmpc2",
        backend="",
        allow_official_only=False,
        seed_list="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
    )

    assert resolved == manifest_path
    assert captured["backend"] == "official_tdmpc2_torch_subprocess"
    assert captured["family"] == "tdmpc2"
    assert captured["mode"] == "proof_compare"
    assert captured["allow_official_only"] is False


def test_parity_proof_run_tdmpc2_without_manifest_blocks_without_aligned_report() -> None:
    result = runner.invoke(
        cli.app,
        [
            "parity",
            "proof-run",
            "--family",
            "tdmpc2",
            "--backend",
            "official_tdmpc2_torch_subprocess",
            "--seed-list",
            ",".join(str(i) for i in range(20)),
        ],
    )
    assert result.exit_code == 2
    assert "tdmpc2_architecture_mismatch_open" in result.stdout


def test_parity_proof_run_tdmpc2_resolves_canonical_backend_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from worldflux.execution import BackendExecutionResult, ManifestResolution

    captured: dict[str, object] = {}

    def _fake_resolve_execution_manifest(request, *, scripts_root, allow_official_only=False):
        captured["backend"] = request.backend
        captured["family"] = request.family
        captured["allow_official_only"] = allow_official_only
        return ManifestResolution(
            manifest_path=None,
            early_result=BackendExecutionResult(
                status="blocked",
                reason_code="tdmpc2_architecture_mismatch_open",
                message="blocked",
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="compare",
                run_id=request.run_id,
                next_action="fix config",
            ),
        )

    monkeypatch.setattr(
        "worldflux.cli._parity.resolve_execution_manifest", _fake_resolve_execution_manifest
    )

    result = runner.invoke(
        cli.app,
        [
            "parity",
            "proof-run",
            "--family",
            "tdmpc2",
            "--seed-list",
            ",".join(str(i) for i in range(20)),
        ],
    )

    assert result.exit_code == 2
    assert captured["family"] == "tdmpc2"
    assert captured["backend"] == "official_tdmpc2_torch_subprocess"


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
    stability = run_root / "stability_report.json"

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
        if script_name == "stability_report.py":
            output = Path(args[args.index("--output") + 1])
            output.write_text(
                json.dumps({"schema_version": "parity.stability.v1", "status": "single_run"}),
                encoding="utf-8",
            )
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
        "stability_report.py",
    ]
    assert equivalence.exists()
    assert markdown.exists()
    assert stability.exists()
    assert "Parity Proof Report" in result.stdout


def test_parity_proof_report_passes_history_reports_to_stability_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run_hist"
    run_root.mkdir(parents=True)
    runs = run_root / "parity_runs.jsonl"
    runs.write_text("", encoding="utf-8")
    history = tmp_path / "history_equivalence_report.json"
    history.write_text("{}", encoding="utf-8")

    captured: dict[str, list[str]] = {}

    def _run(script_name: str, args: list[str]) -> str:
        if script_name == "stats_equivalence.py":
            Path(args[args.index("--output") + 1]).write_text(
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
            Path(args[args.index("--validity-report") + 1]).write_text("{}", encoding="utf-8")
        elif script_name == "validate_matrix_completeness.py":
            Path(args[args.index("--output") + 1]).write_text(
                '{"missing_pairs":0,"pass":true}', encoding="utf-8"
            )
        elif script_name == "report_markdown.py":
            Path(args[args.index("--output") + 1]).write_text("# Proof\n", encoding="utf-8")
        elif script_name == "stability_report.py":
            captured["args"] = list(args)
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"schema_version": "parity.stability.v1", "status": "stable"}),
                encoding="utf-8",
            )
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
            "--history-equivalence-report",
            str(history),
        ],
    )

    assert result.exit_code == 0
    assert "--history-equivalence-report" in captured["args"]
    assert str(history) in captured["args"]


def test_parity_proof_combined_runs_all_phases(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("schema_version: parity.manifest.v1\n", encoding="utf-8")
    output_dir = tmp_path / "proof_combined"
    calls: list[tuple[str, list[str]]] = []

    def _run(script_name: str, args: list[str]) -> str:
        calls.append((script_name, list(args)))
        if script_name == "run_parity_matrix.py":
            run_output_dir = Path(args[args.index("--output-dir") + 1])
            run_output_dir.mkdir(parents=True, exist_ok=True)
            (run_output_dir / "parity_runs.jsonl").write_text("", encoding="utf-8")
            (run_output_dir / "seed_plan.json").write_text(
                '{"seed_values":[0,1]}', encoding="utf-8"
            )
            (run_output_dir / "run_context.json").write_text(
                '{"systems":["official","worldflux"]}',
                encoding="utf-8",
            )
            return "proof-run-ok"
        if script_name == "validate_matrix_completeness.py":
            Path(args[args.index("--output") + 1]).write_text(
                '{"missing_pairs":0,"pass":true}',
                encoding="utf-8",
            )
            return ""
        if script_name == "stats_equivalence.py":
            Path(args[args.index("--output") + 1]).write_text(
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
            Path(args[args.index("--validity-report") + 1]).write_text("{}", encoding="utf-8")
            return ""
        if script_name == "report_markdown.py":
            Path(args[args.index("--output") + 1]).write_text("# Proof\n", encoding="utf-8")
            return ""
        if script_name == "stability_report.py":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"schema_version": "parity.stability.v1", "status": "single_run"}),
                encoding="utf-8",
            )
            return ""
        raise AssertionError(f"unexpected script: {script_name}")

    monkeypatch.setattr(cli, "_run_parity_proof_script", _run)

    result = runner.invoke(
        cli.app,
        [
            "parity",
            "proof",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--seed-list",
            "0,1",
        ],
    )

    assert result.exit_code == 0
    assert [name for name, _args in calls] == [
        "run_parity_matrix.py",
        "validate_matrix_completeness.py",
        "stats_equivalence.py",
        "report_markdown.py",
        "stability_report.py",
    ]
    run_args = calls[0][1]
    assert "--seed-list" in run_args
    coverage_args = calls[1][1]
    assert "--seed-plan" in coverage_args
    assert "--run-context" in coverage_args
    assert (output_dir / "coverage_report.json").exists()
    assert (output_dir / "equivalence_report.json").exists()
    assert (output_dir / "equivalence_report.md").exists()
    assert (output_dir / "stability_report.json").exists()


def test_parity_proof_combined_resolves_manifest_when_omitted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "proof_combined_auto"
    calls: list[tuple[str, list[str]]] = []

    def _run(script_name: str, args: list[str]) -> str:
        calls.append((script_name, list(args)))
        if script_name == "run_parity_matrix.py":
            run_output_dir = Path(args[args.index("--output-dir") + 1])
            run_output_dir.mkdir(parents=True, exist_ok=True)
            (run_output_dir / "parity_runs.jsonl").write_text("", encoding="utf-8")
            return ""
        if script_name == "validate_matrix_completeness.py":
            Path(args[args.index("--output") + 1]).write_text(
                '{"missing_pairs":0,"pass":true}',
                encoding="utf-8",
            )
            return ""
        if script_name == "stats_equivalence.py":
            Path(args[args.index("--output") + 1]).write_text(
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
            Path(args[args.index("--validity-report") + 1]).write_text("{}", encoding="utf-8")
            return ""
        if script_name == "report_markdown.py":
            Path(args[args.index("--output") + 1]).write_text("# Proof\n", encoding="utf-8")
            return ""
        if script_name == "stability_report.py":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"schema_version": "parity.stability.v1", "status": "single_run"}),
                encoding="utf-8",
            )
            return ""
        raise AssertionError(f"unexpected script: {script_name}")

    monkeypatch.setattr(cli, "_run_parity_proof_script", _run)

    result = runner.invoke(
        cli.app,
        [
            "parity",
            "proof",
            "--family",
            "dreamer",
            "--seed-list",
            ",".join(str(i) for i in range(20)),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    run_manifest_arg = calls[0][1][calls[0][1].index("--manifest") + 1]
    assert run_manifest_arg.endswith("official_vs_worldflux_full_v2.yaml")


def test_parity_proof_combined_enforce_exits_on_failed_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("schema_version: parity.manifest.v1\n", encoding="utf-8")
    output_dir = tmp_path / "proof_combined_fail"

    def _run(script_name: str, args: list[str]) -> str:
        if script_name == "run_parity_matrix.py":
            run_output_dir = Path(args[args.index("--output-dir") + 1])
            run_output_dir.mkdir(parents=True, exist_ok=True)
            (run_output_dir / "parity_runs.jsonl").write_text("", encoding="utf-8")
            return ""
        if script_name == "validate_matrix_completeness.py":
            Path(args[args.index("--output") + 1]).write_text(
                '{"missing_pairs":0,"pass":true}',
                encoding="utf-8",
            )
            return ""
        if script_name == "stats_equivalence.py":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps(
                    {
                        "global": {
                            "parity_pass_final": False,
                            "validity_pass": True,
                            "missing_pairs": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )
            Path(args[args.index("--validity-report") + 1]).write_text("{}", encoding="utf-8")
            return ""
        if script_name == "report_markdown.py":
            Path(args[args.index("--output") + 1]).write_text("# Proof\n", encoding="utf-8")
            return ""
        if script_name == "stability_report.py":
            Path(args[args.index("--output") + 1]).write_text(
                json.dumps({"schema_version": "parity.stability.v1", "status": "single_run"}),
                encoding="utf-8",
            )
            return ""
        raise AssertionError(f"unexpected script: {script_name}")

    monkeypatch.setattr(cli, "_run_parity_proof_script", _run)

    result = runner.invoke(
        cli.app,
        [
            "parity",
            "proof",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--enforce",
        ],
    )

    assert result.exit_code == 1
    assert "Verify - Combined Summary" in result.stdout


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


# ---------------------------------------------------------------------------
# train command
# ---------------------------------------------------------------------------


class TestTrainCLI:
    def test_train_command_exists(self) -> None:
        """Verify that `train` is registered as a CLI command."""
        result = runner.invoke(cli.app, ["train", "--help"])
        assert result.exit_code == 0
        assert "train" in result.output.lower()

    def test_train_help(self) -> None:
        result = runner.invoke(cli.app, ["train", "--help"])
        assert result.exit_code == 0
        assert "worldflux.toml" in result.output.lower() or "config" in result.output.lower()

    def test_train_missing_config_fails(self) -> None:
        with runner.isolated_filesystem():
            result = runner.invoke(cli.app, ["train"])
            assert result.exit_code == 1
            assert "Configuration error" in result.output or "not found" in result.output.lower()

    def test_train_with_valid_config(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            """\
project_name = "test-train"
environment = "custom"
model = "dreamer:ci"
model_type = "dreamer"

[architecture]
obs_shape = [3, 64, 64]
action_dim = 6
hidden_dim = 32

[training]
total_steps = 2
batch_size = 2
sequence_length = 10
learning_rate = 3e-4
device = "cpu"
output_dir = "{output_dir}"
""".format(output_dir=str(tmp_path / "outputs")),
            encoding="utf-8",
        )

        result = runner.invoke(
            cli.app,
            ["train", "--config", str(toml_path), "--steps", "2"],
        )
        assert result.exit_code == 0
        assert "Training Complete" in result.output
        assert "Final step" in result.output

    def test_train_steps_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            """\
project_name = "steps-override"
model = "dreamer:ci"

[architecture]
obs_shape = [3, 64, 64]
action_dim = 6

[training]
total_steps = 999999
batch_size = 2
device = "cpu"
output_dir = "{output_dir}"
""".format(output_dir=str(tmp_path / "outputs")),
            encoding="utf-8",
        )

        result = runner.invoke(
            cli.app,
            ["train", "--config", str(toml_path), "--steps", "2"],
        )
        assert result.exit_code == 0
        assert "Training Complete" in result.output

    def test_train_uses_project_scaffold_runtime_when_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from worldflux import Batch

        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            """\
project_name = "scaffold-runtime"
environment = "atari"
model = "dreamer:ci"
model_type = "dreamer"

[architecture]
obs_shape = [3, 64, 64]
action_dim = 6
hidden_dim = 32

[training]
total_steps = 2
batch_size = 2
sequence_length = 5
learning_rate = 3e-4
device = "cpu"
output_dir = "{output_dir}"

[gameplay]
enabled = true
fps = 8
max_frames = 64

[visualization]
enabled = true
host = "127.0.0.1"
port = 8765
refresh_ms = 250
history_max_points = 32
open_browser = false
""".format(output_dir=str(tmp_path / "outputs")),
            encoding="utf-8",
        )
        (tmp_path / "dashboard").mkdir()

        calls = {"build_training_data": 0, "dashboard_started": 0, "target_steps": None}

        class _Provider:
            def sample(self, batch_size: int, seq_len: int | None = None, device: str = "cpu"):
                assert seq_len is not None
                return Batch(
                    obs=torch.randn(batch_size, seq_len, 3, 64, 64, device=device),
                    actions=torch.randn(batch_size, seq_len, 6, device=device),
                    rewards=torch.randn(batch_size, seq_len, device=device),
                    terminations=torch.zeros(batch_size, seq_len, device=device),
                )

        class _FakeDatasetModule:
            @staticmethod
            def build_training_data(
                model_config,
                frame_callback=None,
                phase_callback=None,
            ):
                del model_config, frame_callback, phase_callback
                calls["build_training_data"] += 1
                return _Provider(), (lambda: None), "offline"

        class _FakeMetricBuffer:
            def __init__(self, max_points: int):
                self.max_points = max_points

            def set_gameplay_available(self, available: bool) -> None:
                del available

            def set_phase(self, phase: str, detail: str | None = None) -> None:
                del phase, detail

            def set_status(self, status: str, error: str | None = None) -> None:
                del status, error

            def set_target_steps(self, total_steps: int) -> None:
                calls["target_steps"] = total_steps

        class _FakeGameplayBuffer:
            def __init__(self, max_frames: int, fps: int):
                self.max_frames = max_frames
                self.fps = fps

            def set_phase(self, phase: str, detail: str | None = None) -> None:
                del phase, detail

            def set_status(self, status: str) -> None:
                del status

            def append_frame(self, *args, **kwargs) -> None:
                del args, kwargs

        class _FakeDashboardServer:
            def __init__(self, **kwargs):
                del kwargs
                self.url = "http://127.0.0.1:8765"

            def start(self) -> int:
                calls["dashboard_started"] += 1
                return 8765

            def open_browser(self) -> None:
                return None

            def schedule_shutdown(self, delay_seconds: float) -> None:
                del delay_seconds

            def wait_for_stop(self, timeout: float | None = None) -> bool:
                del timeout
                return True

        class _FakeDashboardCallback:
            def __init__(self, buffer, jsonl_path: Path):
                del buffer, jsonl_path

            def on_train_begin(self, trainer) -> None:
                del trainer

            def on_step_end(self, trainer) -> None:
                del trainer

            def on_train_end(self, trainer) -> None:
                del trainer

            def close(self) -> None:
                return None

        fake_runtime = SimpleNamespace(
            dataset_module=_FakeDatasetModule(),
            dashboard_module=SimpleNamespace(
                MetricBuffer=_FakeMetricBuffer,
                GameplayBuffer=_FakeGameplayBuffer,
                MetricsDashboardServer=_FakeDashboardServer,
                DashboardCallback=_FakeDashboardCallback,
            ),
            dashboard_root=tmp_path / "dashboard",
        )

        monkeypatch.setattr(
            "worldflux.training.data.create_random_buffer",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("fallback random buffer should not be used")
            ),
        )
        monkeypatch.setattr(
            "worldflux.cli._train._load_scaffold_runtime",
            lambda _root: fake_runtime,
        )

        result = runner.invoke(
            cli.app,
            ["train", "--config", str(toml_path), "--steps", "2"],
        )

        assert result.exit_code == 0
        assert calls["build_training_data"] == 1
        assert calls["dashboard_started"] == 1
        assert calls["target_steps"] == 2
        assert "Dashboard:" in result.output
        assert "Training Complete" in result.output

    def test_train_summary_prefers_explicit_quick_verify_next_step(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            """\
project_name = "quick-next-step"
model = "dreamer:ci"

[architecture]
obs_shape = [3, 64, 64]
action_dim = 6

[training]
total_steps = 2
batch_size = 2
device = "cpu"
output_dir = "{output_dir}"

[verify]
mode = "quick"
""".format(output_dir=str(tmp_path / "outputs")),
            encoding="utf-8",
        )

        result = runner.invoke(
            cli.app,
            ["train", "--config", str(toml_path), "--steps", "2"],
        )
        assert result.exit_code == 0
        assert "Next: worldflux verify --target " in result.output
        assert "--mode quick" in result.output

    def test_train_non_native_backend_returns_blocked_exit_code(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            """\
project_name = "blocked-train"
model = "tdmpc2:proof_5m"

[architecture]
obs_shape = [39]
action_dim = 4

[training]
total_steps = 2
batch_size = 2
device = "cpu"
backend = "official_tdmpc2_torch_subprocess"
backend_profile = "proof_5m"
output_dir = "{output_dir}"

[verify]
env = "dmcontrol/walker-run"
""".format(output_dir=str(tmp_path / "outputs")),
            encoding="utf-8",
        )

        result = runner.invoke(cli.app, ["train", "--config", str(toml_path)])
        assert result.exit_code == 2
        assert "blocked" in result.output.lower()

    def test_train_tdmpc2_aligned_backend_remains_blocked(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        report_path = tmp_path / "alignment.json"
        report_path.write_text('{"status":"aligned"}', encoding="utf-8")
        monkeypatch.setenv("WORLDFLUX_TDMPC2_ALIGNMENT_REPORT", str(report_path))

        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            """\
project_name = "aligned-train"
model = "tdmpc2:proof_5m"

[architecture]
obs_shape = [39]
action_dim = 4

[training]
total_steps = 2
batch_size = 2
device = "cpu"
backend = "official_tdmpc2_torch_subprocess"
backend_profile = "proof_5m"
output_dir = "{output_dir}"

[verify]
env = "dmcontrol/walker-run"
""".format(output_dir=str(tmp_path / "outputs")),
            encoding="utf-8",
        )

        result = runner.invoke(cli.app, ["train", "--config", str(toml_path)])
        assert result.exit_code == 2
        assert "Delegated Training Result" in result.output
        assert "official_tdmpc2_torch_subprocess" in result.output
        assert "blocked" in result.output.lower()
        assert "not implemented" in result.output.lower()


# ---------------------------------------------------------------------------
# verify format options
# ---------------------------------------------------------------------------


class TestVerifyFormatOptions:
    def test_verify_json_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from worldflux.verify import VerifyResult

        fake_result = VerifyResult(
            passed=True,
            target="m.pt",
            baseline="official/dreamerv3",
            env="atari/pong",
            demo=True,
            elapsed_seconds=3.1,
            stats={"samples": 500, "mean_drop_ratio": 0.01},
            verdict_reason="Synthetic demo mode: example pass (not proof)",
        )
        monkeypatch.setattr(cli.ParityVerifier, "run", classmethod(lambda cls, **kw: fake_result))
        # Force proof mode
        result = runner.invoke(
            cli.app,
            ["verify", "--target", "m.pt", "--demo", "--format", "json", "--mode", "proof"],
        )
        assert result.exit_code == 0
        output = result.output
        assert '"passed": true' in output or '"passed":true' in output
        assert '"synthetic_provenance"' in output


def test_models_list_human_output_uses_reference_family_label() -> None:
    result = runner.invoke(cli.app, ["models", "list"])
    assert result.exit_code == 0
    assert "reference-family" in result.output
    assert "published evidence bundles" not in result.output
    assert "advanced evidence workflow" not in result.output


def test_models_list_public_surface_mentions_advanced_workflow() -> None:
    result = runner.invoke(cli.app, ["models", "list", "--surface", "public", "--format", "json"])
    assert result.exit_code == 0
    assert "dreamerv3:official_xl" in result.output
    result = runner.invoke(cli.app, ["models", "list", "--surface", "public"])
    assert result.exit_code == 0
    assert "advanced evidence workflow" in result.output


def test_models_list_json_preserves_reference_machine_value() -> None:
    result = runner.invoke(cli.app, ["models", "list", "--format", "json"])
    assert result.exit_code == 0
    assert '"maturity": "reference"' in result.output
    assert '"support_tier": "supported"' in result.output
    assert '"support_tier": "advanced"' not in result.output


def test_models_list_accepts_reference_family_alias() -> None:
    result = runner.invoke(
        cli.app,
        [
            "models",
            "list",
            "--surface",
            "all",
            "--maturity",
            "reference-family",
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert "dreamerv3:size12m" in result.output
    assert "No models match" not in result.output


def test_models_list_all_surface_exposes_experimental_and_internal() -> None:
    result = runner.invoke(cli.app, ["models", "list", "--surface", "all", "--format", "json"])
    assert result.exit_code == 0
    assert '"jepa:base"' in result.output
    assert '"dit:base"' in result.output


def test_models_info_reference_note_is_conservative() -> None:
    result = runner.invoke(cli.app, ["models", "info", "dreamerv3:size12m"])
    assert result.exit_code == 0
    assert "reference-family" in result.output
    assert "not by itself a public" in result.output
    assert "proof claim" in result.output


def test_train_marks_random_fallback_runs_as_contract_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    toml_path = tmp_path / "worldflux.toml"
    output_dir = tmp_path / "outputs"
    toml_path.write_text(
        f"""\
project_name = "fallback-train"
environment = "atari"
model = "dreamer:ci"

[architecture]
obs_shape = [3, 64, 64]
action_dim = 6

[training]
total_steps = 2
batch_size = 2
device = "cpu"
output_dir = "{output_dir}"

[data]
source = "gym"
gym_env = "ALE/Breakout-v5"
""",
        encoding="utf-8",
    )

    class _FailingDatasetModule:
        @staticmethod
        def build_training_data(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("dataset bootstrap failed")

    fake_runtime = SimpleNamespace(
        dataset_module=_FailingDatasetModule(),
        dashboard_module=SimpleNamespace(),
        dashboard_root=tmp_path / "dashboard",
    )
    monkeypatch.setattr("worldflux.cli._train._load_scaffold_runtime", lambda _root: fake_runtime)

    result = runner.invoke(cli.app, ["train", "--config", str(toml_path), "--steps", "2"])

    assert result.exit_code == 0
    assert "degraded" in result.output.lower()
    payload = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert payload["run_classification"] == "contract_smoke"
    assert payload["support_surface"] == "supported"
    assert payload["data_mode"] == "random"
    assert "random_replay_fallback" in payload["degraded_modes"]
    assert "scaffold_runtime_fallback" in payload["degraded_modes"]
