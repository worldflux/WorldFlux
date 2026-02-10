"""Runtime behavior checks for generated scaffold projects."""

from __future__ import annotations

import builtins
import importlib.util
import warnings
from pathlib import Path
from types import ModuleType

from worldflux import create_world_model
from worldflux.scaffold.generator import generate_project


def _context() -> dict[str, object]:
    return {
        "project_name": "runtime-demo",
        "environment": "atari",
        "model": "dreamer:ci",
        "model_type": "dreamer",
        "obs_shape": [3, 64, 64],
        "action_dim": 6,
        "hidden_dim": 32,
        "device": "cpu",
    }


def _import_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_dataset_falls_back_without_userwarning_when_ale_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "generated"
    generate_project(target, _context(), force=False)

    dataset_module = _import_module("generated_dataset", target / "dataset.py")

    model = create_world_model(
        model="dreamer:ci",
        obs_shape=(3, 64, 64),
        action_dim=6,
        device="cpu",
    )

    monkeypatch.chdir(target)

    original_import = builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name == "ale_py":
            raise ModuleNotFoundError(name="ale_py")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data_source, cleanup, mode = dataset_module.build_training_data(model.config)

    assert mode == "offline"
    assert not any(issubclass(item.category, UserWarning) for item in caught)
    cleanup()
    del data_source


def test_generated_train_template_uses_training_phase_with_gameplay_unavailable(
    tmp_path: Path,
) -> None:
    target = tmp_path / "generated"
    generate_project(target, _context(), force=False)

    content = (target / "train.py").read_text(encoding="utf-8")
    assert 'publish_phase("training"' in content
    assert "Live gameplay unavailable" in content
    assert 'publish_phase("unavailable", unavailable_detail)' not in content
    assert "except KeyboardInterrupt:" in content
    assert "Interrupted while waiting for dashboard shutdown." in content
