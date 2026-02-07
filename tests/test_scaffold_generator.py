"""Tests for project scaffolding generator."""

from __future__ import annotations

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
) -> dict[str, object]:
    return {
        "project_name": project_name,
        "environment": environment,
        "model": model,
        "model_type": model_type,
        "obs_shape": obs_shape or [3, 64, 64],
        "action_dim": action_dim,
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
    assert 'source = "gym"' in toml_content
    assert "[gameplay]\nenabled = true" in toml_content
    assert "[online_collection]\nenabled = true" in toml_content

    readme_content = (target / "README.md").read_text(encoding="utf-8")
    assert "uv run python train.py" in readme_content
    assert "uv run python inference.py" in readme_content
    assert "Dashboard:" in readme_content


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
            device="cpu",
        ),
        force=True,
    )

    toml_content = (target / "worldflux.toml").read_text(encoding="utf-8")
    assert 'project_name = "second"' in toml_content
    assert 'model = "tdmpc2:ci"' in toml_content
    assert "obs_shape = [39]" in toml_content
    assert "action_dim = 4" in toml_content
    assert 'source = "random"' in toml_content
    assert "[gameplay]\nenabled = false" in toml_content
    assert "[online_collection]\nenabled = false" in toml_content


def test_generate_project_rejects_file_target(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_text("file", encoding="utf-8")

    with pytest.raises(FileExistsError, match="is a file"):
        generate_project(target, _context(), force=False)
