"""Tests for backend-neutral parity contracts and adapters."""

from __future__ import annotations

from pathlib import Path

from worldflux.parity import (
    DreamerOfficialJAXSubprocessAdapter,
    DreamerWorldFluxJAXSubprocessAdapter,
    discover_artifacts,
    get_backend_adapter_registry,
    stable_recipe_hash,
)


def test_stable_recipe_hash_is_deterministic() -> None:
    recipe_a = {"steps": 10, "batch_size": 2}
    recipe_b = {"batch_size": 2, "steps": 10}
    assert stable_recipe_hash(recipe_a) == stable_recipe_hash(recipe_b)


def test_backend_adapter_registry_contains_dreamer_adapter() -> None:
    registry = get_backend_adapter_registry()
    adapter = registry.require("official_dreamerv3_jax_subprocess")
    assert isinstance(adapter, DreamerOfficialJAXSubprocessAdapter)


def test_backend_adapter_registry_contains_worldflux_dreamer_jax_adapter() -> None:
    registry = get_backend_adapter_registry()
    adapter = registry.require("worldflux_dreamerv3_jax_subprocess")
    assert isinstance(adapter, DreamerWorldFluxJAXSubprocessAdapter)
    assert adapter.backend_kind == "jax_subprocess"


def test_backend_adapter_registry_contains_tdmpc2_adapter() -> None:
    registry = get_backend_adapter_registry()
    adapter = registry.require("official_tdmpc2_torch_subprocess")
    assert adapter.adapter_id == "official_tdmpc2_torch_subprocess"
    assert adapter.backend_kind == "torch_subprocess"


def test_dreamer_adapter_prepare_run_matches_expected_command(tmp_path: Path) -> None:
    adapter = DreamerOfficialJAXSubprocessAdapter()
    spec = adapter.prepare_run(
        recipe={"steps": 110000, "train_ratio": 256},
        env_spec={"task_id": "atari100k_pong"},
        seed=0,
        run_dir=tmp_path / "run",
        repo_root=Path("/tmp/dreamer"),
        python_executable="python3",
        device="cuda",
    )
    assert spec.adapter_id == "official_dreamerv3_jax_subprocess"
    assert spec.backend_kind == "jax_subprocess"
    assert "--configs" in spec.command
    assert "atari100k" in spec.command
    assert "--run.steps" in spec.command
    assert "110000" in spec.command


def test_worldflux_dreamer_jax_adapter_prepare_run_invokes_wrapper(tmp_path: Path) -> None:
    adapter = DreamerWorldFluxJAXSubprocessAdapter()
    spec = adapter.prepare_run(
        recipe={"steps": 110000, "train_ratio": 256},
        env_spec={"task_id": "atari100k_pong"},
        seed=0,
        run_dir=tmp_path / "run",
        repo_root=Path("/tmp/worldflux"),
        python_executable="python3",
        device="cuda",
    )
    assert spec.adapter_id == "worldflux_dreamerv3_jax_subprocess"
    assert spec.backend_kind == "jax_subprocess"
    assert any("worldflux_dreamerv3_jax.py" in part for part in spec.command)


def test_discover_artifacts_collects_expected_paths(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    ckpt = run_root / "dreamerv3_logdir" / "ckpt" / "latest" / "agent.pkl"
    score = run_root / "dreamerv3_logdir" / "scores.jsonl"
    metrics = run_root / "metrics.json"
    config = run_root / "dreamerv3_logdir" / "config.yaml"
    for path in (ckpt, score, metrics, config):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")

    manifest = discover_artifacts(
        run_root=run_root,
        backend_kind="jax_subprocess",
        adapter_id="official_dreamerv3_jax_subprocess",
        recipe_hash="abc",
        command_argv=["python3", "dreamerv3/main.py"],
        source_commit="deadbeef",
        eval_protocol_hash="123",
    )
    assert manifest.config_snapshot is not None
    assert any(path.endswith("agent.pkl") for path in manifest.checkpoint_paths)
    assert any(path.endswith("scores.jsonl") for path in manifest.score_paths)
    assert any(path.endswith("metrics.json") for path in manifest.metrics_paths)


def test_dreamer_adapter_monitor_run_detects_latest_checkpoint(tmp_path: Path) -> None:
    adapter = DreamerOfficialJAXSubprocessAdapter()
    run_root = tmp_path / "run"
    latest_dir = run_root / "dreamerv3_logdir" / "ckpt" / "20260101T000000F000001"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "dreamerv3_logdir" / "config.yaml").write_text("x", encoding="utf-8")
    (run_root / "dreamerv3_logdir" / "scores.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    (run_root / "dreamerv3_logdir" / "ckpt" / "latest").write_text(
        latest_dir.name, encoding="utf-8"
    )
    (latest_dir / "agent.pkl").write_text("x", encoding="utf-8")

    status = adapter.monitor_run(run_dir=run_root)
    assert status["config_present"] is True
    assert status["scores_present"] is True
    assert status["scores_lines"] == 2
    assert status["agent_present"] is True


def test_tdmpc2_adapter_monitor_run_detects_eval_and_checkpoint(tmp_path: Path) -> None:
    adapter = get_backend_adapter_registry().require("official_tdmpc2_torch_subprocess")
    run_root = tmp_path / "run"
    eval_csv = run_root / "outputs" / "eval.csv"
    checkpoint = run_root / "outputs" / "agent.pt"
    eval_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_csv.write_text("step,reward\n1,1.0\n", encoding="utf-8")
    checkpoint.write_text("x", encoding="utf-8")

    status = adapter.monitor_run(run_dir=run_root)
    assert status["eval_csv_present"] is True
    assert status["checkpoint_present"] is True
    assert status["eval_csv_path"].endswith("eval.csv")
    assert status["checkpoint_path"].endswith(".pt")
