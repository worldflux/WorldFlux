"""Tests for project scaffolding generator."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from worldflux.scaffold.generator import generate_project


def _context(
    *,
    project_name: str = "my-world-model",
    environment: str = "atari",
    model: str = "dreamer:ci",
    model_type: str = "dreamer",
    obs_shape: list[int] | None = None,
    action_dim: int = 6,
    hidden_dim: int = 32,
    device: str = "cpu",
    training_total_steps: int = 100000,
    training_batch_size: int = 16,
) -> dict[str, object]:
    return {
        "project_name": project_name,
        "environment": environment,
        "model": model,
        "model_type": model_type,
        "obs_shape": obs_shape or [3, 64, 64],
        "action_dim": action_dim,
        "training_total_steps": training_total_steps,
        "training_batch_size": training_batch_size,
        "hidden_dim": hidden_dim,
        "device": device,
    }


def test_generate_project_creates_expected_files(tmp_path: Path) -> None:
    target = tmp_path / "demo"
    created_files = generate_project(target, _context(), force=False)

    expected_files = {
        "worldflux.toml",
        "train.py",
        "inference.py",
        "dataset.py",
        "local_dashboard.py",
        "dashboard/index.html",
        "README.md",
    }
    assert {str(path.relative_to(target)) for path in created_files} == expected_files

    for relative_path in expected_files:
        assert (target / relative_path).exists()

    toml_content = (target / "worldflux.toml").read_text(encoding="utf-8")
    assert 'project_name = "my-world-model"' in toml_content
    assert 'model = "dreamer:ci"' in toml_content
    assert "obs_shape = [3, 64, 64]" in toml_content
    assert "total_steps = 100000" in toml_content
    assert "batch_size = 16" in toml_content
    assert 'source = "gym"' in toml_content
    assert "[gameplay]\nenabled = true" in toml_content
    assert "[online_collection]\nenabled = true" in toml_content

    readme_content = (target / "README.md").read_text(encoding="utf-8")
    assert "uv run python train.py" in readme_content
    assert "uv run python inference.py" in readme_content
    assert "Dashboard:" in readme_content

    dataset_content = (target / "dataset.py").read_text(encoding="utf-8")
    assert "warnings.warn(" not in dataset_content
    assert 'print(f"[dataset] {message}")' in dataset_content

    train_content = (target / "train.py").read_text(encoding="utf-8")
    assert "dashboard_buffer.set_target_steps(total_steps)" in train_content
    assert "Live gameplay unavailable" in train_content
    assert "Interrupted while waiting for dashboard shutdown." in train_content
    assert "except KeyboardInterrupt:" in train_content

    inference_content = (target / "inference.py").read_text(encoding="utf-8")
    assert (
        "def _unwrap_model_state_dict(payload: object) -> dict[str, torch.Tensor]:"
        in inference_content
    )
    assert 'payload.get("model_state_dict")' in inference_content

    dashboard_backend = (target / "local_dashboard.py").read_text(encoding="utf-8")
    assert "def set_target_steps(self, total_steps: int) -> None:" in dashboard_backend
    assert '"progress_percent": progress_percent' in dashboard_backend
    assert '"target_steps": target_steps' in dashboard_backend

    dashboard_frontend = (target / "dashboard/index.html").read_text(encoding="utf-8")
    assert 'data-metric="progress"' in dashboard_frontend
    assert 'id="progress-fill"' in dashboard_frontend
    assert "resolveProgressPercent" in dashboard_frontend


def test_generate_project_rejects_non_empty_directory_without_force(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        generate_project(target, _context(), force=False)


def test_generate_project_overwrites_when_force_enabled(tmp_path: Path) -> None:
    target = tmp_path / "force-project"
    generate_project(target, _context(project_name="first"), force=False)

    generate_project(
        target,
        _context(
            project_name="second",
            environment="mujoco",
            model="tdmpc2:ci",
            model_type="tdmpc2",
            obs_shape=[39],
            action_dim=4,
            training_total_steps=75000,
            training_batch_size=24,
            device="cpu",
        ),
        force=True,
    )

    toml_content = (target / "worldflux.toml").read_text(encoding="utf-8")
    assert 'project_name = "second"' in toml_content
    assert 'model = "tdmpc2:ci"' in toml_content
    assert "obs_shape = [39]" in toml_content
    assert "action_dim = 4" in toml_content
    assert "total_steps = 75000" in toml_content
    assert "batch_size = 24" in toml_content
    assert 'source = "gym"' in toml_content
    assert "[gameplay]\nenabled = true" in toml_content
    assert "[online_collection]\nenabled = true" in toml_content


def test_generate_project_rejects_file_target(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_text("file", encoding="utf-8")

    with pytest.raises(FileExistsError, match="is a file"):
        generate_project(target, _context(), force=False)


def test_generate_project_falls_back_when_pydantic_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pydantic":
            raise ModuleNotFoundError("pydantic missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    target = tmp_path / "no-pydantic"
    created = generate_project(target, _context(), force=False)
    assert created


def test_generate_project_rejects_schema_coercion_when_pydantic_available(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pydantic")
    context = _context()
    context["action_dim"] = "6"  # type: ignore[assignment]

    with pytest.raises(ValueError, match="Invalid scaffold context schema"):
        generate_project(tmp_path / "invalid-schema", context, force=False)


def test_generate_project_rejects_non_positive_training_total_steps(tmp_path: Path) -> None:
    context = _context(training_total_steps=0)
    with pytest.raises(ValueError, match="training_total_steps must be positive"):
        generate_project(tmp_path / "invalid-total-steps", context, force=False)


def test_generate_project_rejects_non_positive_training_batch_size(tmp_path: Path) -> None:
    context = _context(training_batch_size=0)
    with pytest.raises(ValueError, match="training_batch_size must be positive"):
        generate_project(tmp_path / "invalid-batch-size", context, force=False)
