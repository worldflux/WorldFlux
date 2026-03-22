# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for tiered quick verification."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
import torch

from worldflux import create_world_model
from worldflux.training import Trainer, TrainingConfig
from worldflux.verify.quick import quick_verify


class _MiniModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"obs_shape": (4,), "action_dim": 2})()

    def encode(self, obs):
        from worldflux.core.state import State

        return State(tensors={"latent": obs.float()})

    def rollout(self, state, actions):
        from worldflux.core.trajectory import Trajectory

        rewards = actions.sum(dim=-1)
        return Trajectory(
            states=[state] * (actions.shape[0] + 1),
            actions=actions,
            rewards=rewards,
            continues=None,
        )


def test_quick_verify_normalizes_deprecated_offline_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "model"
    target.mkdir()

    monkeypatch.setattr(
        "worldflux.verify.quick._load_model_from_target",
        lambda target_path, device: _MiniModel(),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = quick_verify(str(target), env="atari/pong", tier="offline", episodes=3, horizon=4)

    assert result.protocol_version
    assert result.stats["verification_tier_requested"] == "offline"
    assert result.stats["verification_tier_effective"] == "synthetic"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_quick_verify_normalizes_real_env_smoke_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "model"
    target.mkdir()

    monkeypatch.setattr(
        "worldflux.verify.quick._load_model_from_target",
        lambda target_path, device: _MiniModel(),
    )

    with pytest.warns(DeprecationWarning, match="synthetic"):
        result = quick_verify(
            str(target),
            env="atari/pong",
            tier="real_env_smoke",
            episodes=2,
            horizon=4,
        )

    assert result.stats["verification_tier_requested"] == "real_env_smoke"
    assert result.stats["verification_tier_effective"] == "synthetic"


def test_quick_verify_rejects_unknown_tier(tmp_path: Path) -> None:
    target = tmp_path / "model"
    target.mkdir()

    with pytest.raises(ValueError, match="Unknown verification tier"):
        quick_verify(str(target), tier="unknown")


@pytest.mark.parametrize(
    ("model_id", "kwargs"),
    (
        ("tdmpc2:5m", {"obs_shape": (39,), "action_dim": 6}),
        (
            "dreamerv3:size12m",
            {
                "obs_shape": (8,),
                "action_dim": 2,
                "encoder_type": "mlp",
                "decoder_type": "mlp",
            },
        ),
    ),
)
def test_load_model_from_target_restores_non_ci_save_pretrained(
    tmp_path: Path,
    model_id: str,
    kwargs: dict[str, object],
) -> None:
    from worldflux.verify.quick import _load_model_from_target

    model = create_world_model(model_id, device="cpu", **kwargs)
    target = tmp_path / "saved-model"
    model.save_pretrained(str(target))

    restored = _load_model_from_target(target, device="cpu")

    assert restored.config.model_type == model.config.model_type
    assert restored.config.model_name == model.config.model_name


@pytest.mark.parametrize(
    ("model_id", "kwargs"),
    (
        ("tdmpc2:5m", {"obs_shape": (39,), "action_dim": 6}),
        (
            "dreamerv3:size12m",
            {
                "obs_shape": (8,),
                "action_dim": 2,
                "encoder_type": "mlp",
                "decoder_type": "mlp",
            },
        ),
    ),
)
def test_load_model_from_target_restores_trainer_checkpoint_with_exact_config(
    tmp_path: Path,
    model_id: str,
    kwargs: dict[str, object],
) -> None:
    from worldflux.verify.quick import _load_model_from_target

    model = create_world_model(model_id, device="cpu", **kwargs)
    trainer = Trainer(
        model,
        TrainingConfig(
            total_steps=1,
            batch_size=2,
            sequence_length=2,
            output_dir=str(tmp_path / "outputs"),
            device="cpu",
            auto_quality_check=False,
        ),
        callbacks=[],
    )

    checkpoint = tmp_path / "outputs" / "checkpoint_final.pt"
    trainer.save_checkpoint(str(checkpoint))
    restored = _load_model_from_target(checkpoint, device="cpu")

    assert restored.config.model_type == model.config.model_type
    assert restored.config.model_name == model.config.model_name


def test_build_model_from_config_payload_loads_builtins_before_registry_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _load_builtin_models runs before any ConfigRegistry access.

    The CI e2e failure showed that in a cold process, the ConfigRegistry
    is empty when _build_model_from_config_payload tries to look up the
    config class, causing WorldModelConfig to be used for DreamerV3 payloads
    and raising TypeError on model-specific fields like stoch_discrete.
    """
    from worldflux.core.registry import WorldModelRegistry
    from worldflux.verify.quick import _build_model_from_config_payload

    call_log: list[str] = []
    real_load = WorldModelRegistry._load_builtin_models.__func__  # type: ignore[attr-defined]

    @classmethod  # type: ignore[misc]
    def _tracked_load(cls: type) -> None:
        call_log.append("load_builtin_models")
        real_load(cls)

    monkeypatch.setattr(WorldModelRegistry, "_load_builtin_models", _tracked_load)

    model = create_world_model(
        "dreamerv3:size12m",
        obs_shape=(8,),
        action_dim=2,
        encoder_type="mlp",
        decoder_type="mlp",
        device="cpu",
    )
    _build_model_from_config_payload(model.config.to_dict(), device="cpu", require_model_name=True)
    assert (
        "load_builtin_models" in call_log
    ), "_load_builtin_models must be called before registry lookups"


def test_checkpoint_verify_round_trip_in_fresh_process(tmp_path: Path) -> None:
    """Run checkpoint save + verify in a subprocess to catch cold-registry bugs.

    This test exercises the same code path as the CI e2e-pip-flow job:
    a fresh Python process where WorldModelRegistry has not been populated
    by prior imports.  The bug that prompted this test caused
    WorldModelConfig to receive DreamerV3-specific kwargs (stoch_discrete)
    because _load_builtin_models was called after the config lookup.
    """
    script = textwrap.dedent("""\
        import sys
        from pathlib import Path
        from worldflux import create_world_model
        from worldflux.training import Trainer, TrainingConfig
        from worldflux.verify.quick import _load_model_from_target

        tmp = Path(sys.argv[1])
        model = create_world_model(
            "dreamerv3:size12m", obs_shape=(8,), action_dim=2,
            encoder_type="mlp", decoder_type="mlp", device="cpu",
        )
        trainer = Trainer(
            model,
            TrainingConfig(
                total_steps=1, batch_size=2, sequence_length=2,
                output_dir=str(tmp / "outputs"), device="cpu",
                auto_quality_check=False,
            ),
            callbacks=[],
        )
        checkpoint = tmp / "outputs" / "checkpoint_final.pt"
        trainer.save_checkpoint(str(checkpoint))
        restored = _load_model_from_target(checkpoint, device="cpu")
        assert restored.config.model_type == "dreamer", restored.config.model_type
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"Cold-registry checkpoint round-trip failed:\n{result.stderr}"


def test_load_model_from_target_rejects_legacy_checkpoint_without_model_name(
    tmp_path: Path,
) -> None:
    from worldflux.verify.quick import _load_model_from_target

    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": {},
            "model_config": {"model_type": "tdmpc2"},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="model_name"):
        _load_model_from_target(checkpoint, device="cpu")
