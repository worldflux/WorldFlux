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


def _mujoco_context() -> dict[str, object]:
    return {
        "project_name": "mujoco-demo",
        "environment": "mujoco",
        "model": "tdmpc2:ci",
        "model_type": "tdmpc2",
        "obs_shape": [17],
        "action_dim": 6,
        "hidden_dim": 32,
        "device": "cpu",
    }


def test_mujoco_scaffold_falls_back_to_random_when_gymnasium_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "mujoco-gen"
    generate_project(target, _mujoco_context(), force=False)

    dataset_module = _import_module("mujoco_dataset", target / "dataset.py")

    model = create_world_model(
        model="tdmpc2:ci",
        obs_shape=(17,),
        action_dim=6,
        device="cpu",
    )

    monkeypatch.chdir(target)

    original_import = builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name == "gymnasium":
            raise ModuleNotFoundError(name="gymnasium")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        data_source, cleanup, mode = dataset_module.build_training_data(model.config)

    assert mode in ("offline", "random")
    cleanup()
    del data_source


def test_mujoco_online_provider_emits_frames(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "mujoco-frames"
    generate_project(target, _mujoco_context(), force=False)

    dataset_module = _import_module("mujoco_frames_dataset", target / "dataset.py")

    monkeypatch.chdir(target)

    frames_received: list[tuple] = []

    def _frame_cb(frame, episode, step, reward, done):
        frames_received.append((frame, episode, step, reward, done))

    phases_received: list[tuple] = []

    def _phase_cb(phase, detail=None):
        phases_received.append((phase, detail))

    obs_dim = 17
    action_dim = 6
    capacity = 200
    warmup = 50

    # Build a mock gym environment for MuJoCo
    import types

    import numpy as np

    class FakeMuJoCoEnv:
        class ActionSpace:
            @staticmethod
            def sample():
                return np.random.randn(action_dim).astype(np.float32)

        action_space = ActionSpace()

        def __init__(self):
            self._step_count = 0

        def reset(self, **kwargs):
            self._step_count = 0
            return np.zeros(obs_dim, dtype=np.float32), {}

        def step(self, action):
            self._step_count += 1
            obs = np.random.randn(obs_dim).astype(np.float32)
            done = self._step_count >= 10
            return obs, 1.0, done, False, {}

        def render(self):
            return np.zeros((64, 64, 3), dtype=np.uint8)

        def close(self):
            pass

    original_import = builtins.__import__

    def _import(name: str, *args, **kwargs):
        if name == "gymnasium":
            mod = types.ModuleType("gymnasium")
            mod.make = lambda env_name, **kw: FakeMuJoCoEnv()
            return mod
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    provider = dataset_module.OnlineMujocoBatchProvider(
        gym_env="HalfCheetah-v5",
        obs_shape=(obs_dim,),
        action_dim=action_dim,
        capacity=capacity,
        warmup_transitions=warmup,
        collect_steps_per_update=8,
        max_episode_steps=10,
        frame_callback=_frame_cb,
        phase_callback=_phase_cb,
        frame_fps=30,
    )

    assert len(frames_received) > 0, "Expected frame callbacks during warmup"
    assert any(p[0] == "collecting" for p in phases_received)

    # Verify frame data is an ndarray with 3 channels
    first_frame = frames_received[0][0]
    assert hasattr(first_frame, "shape")
    assert len(first_frame.shape) == 3
    assert first_frame.shape[2] == 3

    provider.close()
