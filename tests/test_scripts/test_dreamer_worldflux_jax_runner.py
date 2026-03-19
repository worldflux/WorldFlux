# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for the WorldFlux Dreamer JAX proof runner."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module(name: str, relative: str):
    script_path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_command_uses_worldflux_launcher_with_required_flags(tmp_path: Path) -> None:
    mod = _load_module(
        "dreamer_worldflux_jax_runner", "scripts/parity/runtime/dreamer_worldflux_jax_runner.py"
    )

    class _Args:
        repo_root = tmp_path / "worldflux"
        official_repo_root = None
        run_dir = tmp_path / "run"
        task_id = "atari100k_pong"
        seed = 7
        steps = 110000
        device = "cpu"
        python_executable = "python3"

    expected_vendor_root = _Args.repo_root / "third_party" / "dreamerv3_official"
    resolved = mod._resolve_official_repo_root(_Args())
    assert resolved == expected_vendor_root.resolve()

    command = mod._build_command(_Args())
    assert command[:3] == [
        "python3",
        "-m",
        "worldflux.backends.jax.dreamerv3.launcher",
    ]
    assert str(expected_vendor_root) in command
    assert str(expected_vendor_root / "dreamerv3" / "main.py") not in command
    assert "--logdir" in command
    assert "--task" in command
    assert "atari100k_pong" in command
    assert "--seed" in command
    assert "7" in command
    assert "--run.steps" in command
    assert "110000" in command


def test_worldflux_launcher_forwards_args_to_dreamer_module(monkeypatch, tmp_path: Path) -> None:
    from worldflux.backends.jax.dreamerv3 import launcher

    captured: dict[str, object] = {}
    vendor_root = tmp_path / "dreamerv3_official"
    vendor_root.mkdir(parents=True, exist_ok=True)

    class _FakeMainModule:
        @staticmethod
        def main(argv=None):
            captured["argv"] = list(argv or [])
            return 0

    monkeypatch.setattr(
        launcher,
        "_import_dreamer_main_module",
        lambda official_repo_root: _FakeMainModule,
    )

    exit_code = launcher.main(
        [
            "--official-repo-root",
            str(vendor_root),
            "--logdir",
            str(tmp_path / "run" / "dreamerv3_logdir"),
            "--task",
            "atari100k_pong",
            "--seed",
            "7",
        ]
    )

    assert exit_code == 0
    assert captured["argv"] == [
        "--logdir",
        str(tmp_path / "run" / "dreamerv3_logdir"),
        "--task",
        "atari100k_pong",
        "--seed",
        "7",
    ]


def test_runtime_reports_missing_required_artifacts(tmp_path: Path) -> None:
    from worldflux.backends.jax.dreamerv3.runtime import missing_required_artifacts

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    missing = missing_required_artifacts(run_dir)

    assert missing == [
        "config_yaml",
        "scores_jsonl",
        "metrics_jsonl",
        "latest_pointer",
        "agent_pkl",
        "replay_pkl",
        "step_pkl",
        "done_marker",
    ]


def test_runtime_accepts_complete_required_artifacts(tmp_path: Path) -> None:
    from worldflux.backends.jax.dreamerv3.runtime import missing_required_artifacts

    run_dir = tmp_path / "run"
    latest_dir = run_dir / "dreamerv3_logdir" / "ckpt" / "20260101T000000F000001"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dreamerv3_logdir" / "config.yaml").write_text("x", encoding="utf-8")
    (run_dir / "dreamerv3_logdir" / "scores.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "dreamerv3_logdir" / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "dreamerv3_logdir" / "ckpt" / "latest").write_text(latest_dir.name, encoding="utf-8")
    (latest_dir / "agent.pkl").write_text("x", encoding="utf-8")
    (latest_dir / "replay.pkl").write_text("x", encoding="utf-8")
    (latest_dir / "step.pkl").write_text("x", encoding="utf-8")
    (latest_dir / "done").write_text("", encoding="utf-8")

    assert missing_required_artifacts(run_dir) == []


def test_runtime_builds_proof_metadata_with_expected_fields() -> None:
    from worldflux.backends.jax.dreamerv3.runtime import build_proof_metadata

    metadata = build_proof_metadata(
        task_id="atari100k_pong",
        seed=7,
        device="cpu",
        steps=110000,
        eval_interval=5000,
        eval_episodes=1,
        eval_window=10,
        recipe_hash="recipe-hash",
        backend_kind="jax_subprocess",
        adapter_id="worldflux_dreamerv3_jax_subprocess",
        artifact_manifest={"adapter_id": "worldflux_dreamerv3_jax_subprocess"},
        command=["python3", "-m", "worldflux.backends.jax.dreamerv3.launcher"],
        repo_root=Path("/tmp/worldflux"),
        scores_file=Path("/tmp/worldflux/run/dreamerv3_logdir/scores.jsonl"),
        stdout_tail="out",
        stderr_tail="err",
    )

    assert metadata["mode"] == "worldflux_jax"
    assert metadata["backend_kind"] == "jax_subprocess"
    assert metadata["adapter_id"] == "worldflux_dreamerv3_jax_subprocess"
    assert metadata["recipe_hash"] == "recipe-hash"
    assert metadata["model_id"] == "dreamerv3:official_xl"
    assert metadata["model_profile"] == "official_xl"
    assert metadata["strict_official_semantics"] is True
    assert metadata["policy_mode"] == "parity_candidate"
    assert metadata["policy_impl"] == "candidate_actor_stateful"
    assert metadata["train_budget"]["steps"] == 110000
    assert metadata["eval_protocol"]["eval_interval"] == 5000
    assert metadata["command_source"] == "worldflux_launcher_module"
    assert metadata["implementation_source"] == "worldflux_backends_jax_dreamerv3"


def test_runner_discovers_scores_from_run_dir_only(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module(
        "dreamer_worldflux_jax_runner_runtime",
        "scripts/parity/runtime/dreamer_worldflux_jax_runner.py",
    )
    repo_root = tmp_path / "repo"
    official_root = repo_root / "third_party" / "dreamerv3_official"
    run_dir = tmp_path / "run"
    metrics_out = tmp_path / "metrics.json"
    scores_path = run_dir / "dreamerv3_logdir" / "scores.jsonl"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text("{}\n", encoding="utf-8")
    official_root.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        mod,
        "_parse_args",
        lambda: SimpleNamespace(
            repo_root=repo_root,
            official_repo_root=official_root,
            task_id="atari100k_pong",
            seed=7,
            steps=110000,
            device="cpu",
            run_dir=run_dir,
            metrics_out=metrics_out,
            eval_window=10,
            eval_interval=5000,
            eval_episodes=1,
            timeout_sec=0,
            scores_file=None,
            python_executable="python3",
        ),
    )
    monkeypatch.setattr(
        mod,
        "run_command",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(mod, "validate_required_artifacts", lambda run_dir: {})

    def _find_latest_file(search_roots, candidate_names):
        captured["search_roots"] = [Path(root) for root in search_roots]
        captured["candidate_names"] = list(candidate_names)
        return scores_path

    monkeypatch.setattr(mod, "find_latest_file", _find_latest_file)
    monkeypatch.setattr(mod, "load_jsonl_curve", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(mod, "curve_final_mean", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(mod, "curve_auc", lambda *args, **kwargs: 2.0)

    class _FakeManifest:
        backend_kind = "jax_subprocess"
        adapter_id = "worldflux_dreamerv3_jax_subprocess"
        recipe_hash = "recipe-hash"

        @staticmethod
        def to_dict():
            return {"adapter_id": "worldflux_dreamerv3_jax_subprocess"}

    class _FakeRegistry:
        @staticmethod
        def require(adapter_id: str):
            assert adapter_id == "worldflux_dreamerv3_jax_subprocess"
            return SimpleNamespace(collect_artifacts=lambda **kwargs: _FakeManifest())

    monkeypatch.setattr(mod, "get_backend_adapter_registry", lambda: _FakeRegistry())

    def _write_metrics(**kwargs):
        payload = {"metadata": kwargs["metadata"], "metrics_out": str(kwargs["metrics_out"])}
        captured["payload"] = payload
        return payload

    monkeypatch.setattr(mod, "write_metrics", _write_metrics)

    exit_code = mod.main()

    assert exit_code == 0
    assert captured["search_roots"] == [run_dir.resolve()]
    assert captured["candidate_names"] == ["scores.jsonl", "metrics.jsonl"]
    metadata = captured["payload"]["metadata"]
    assert metadata["command_source"] == "worldflux_launcher_module"
    assert metadata["implementation_source"] == "worldflux_backends_jax_dreamerv3"
    assert "official_repo_root" not in metadata
    assert metadata["strict_official_semantics"] is True
    assert metadata["policy_mode"] == "parity_candidate"
    assert metadata["policy_impl"] == "candidate_actor_stateful"
    assert metadata["train_budget"]["steps"] == 110000
    assert metadata["train_budget"]["train_ratio"] == 256.0
    assert metadata["eval_protocol"] == {
        "eval_interval": 5000,
        "eval_episodes": 1,
        "eval_window": 10,
        "environment_backend": "gymnasium",
        "log_every": 120,
        "report_every": 300,
        "save_every": 900,
    }


def test_runner_fails_when_required_artifacts_are_missing(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module(
        "dreamer_worldflux_jax_runner_missing",
        "scripts/parity/runtime/dreamer_worldflux_jax_runner.py",
    )
    repo_root = tmp_path / "repo"
    official_root = repo_root / "third_party" / "dreamerv3_official"
    run_dir = tmp_path / "run"
    metrics_out = tmp_path / "metrics.json"
    (run_dir / "dreamerv3_logdir" / "scores.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "dreamerv3_logdir" / "scores.jsonl").write_text("{}\n", encoding="utf-8")
    official_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        mod,
        "_parse_args",
        lambda: SimpleNamespace(
            repo_root=repo_root,
            official_repo_root=official_root,
            task_id="atari100k_pong",
            seed=7,
            steps=110000,
            device="cpu",
            run_dir=run_dir,
            metrics_out=metrics_out,
            eval_window=10,
            eval_interval=5000,
            eval_episodes=1,
            timeout_sec=0,
            scores_file=None,
            python_executable="python3",
        ),
    )
    monkeypatch.setattr(
        mod,
        "run_command",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    try:
        mod.main()
    except SystemExit as exc:
        assert "missing required Dreamer artifacts" in str(exc)
        assert "config_yaml" in str(exc)
        assert "metrics_jsonl" in str(exc)
    else:
        raise AssertionError("expected SystemExit for missing required artifacts")


def test_vendor_checkout_exists_in_repo() -> None:
    root = Path(__file__).resolve().parents[2]
    vendor_root = root / "third_party" / "dreamerv3_official"
    assert (vendor_root / "dreamerv3" / "main.py").exists()
    assert (vendor_root / "dreamerv3" / "agent.py").exists()
    assert (vendor_root / "embodied" / "__init__.py").exists()
    assert (vendor_root / "LICENSE").exists()


def test_vendored_dreamer_debug_dummy_cpu_smoke_runs_when_jax_available(tmp_path: Path) -> None:
    if sys.version_info >= (3, 13):
        return
    try:
        import jax  # noqa: F401
    except ImportError:
        return

    root = Path(__file__).resolve().parents[2]
    vendor_root = root / "third_party" / "dreamerv3_official"
    logdir = tmp_path / "smoke"
    cmd = [
        sys.executable,
        str(vendor_root / "dreamerv3" / "main.py"),
        "--configs",
        "debug",
        "--task",
        "dummy_disc",
        "--run.steps",
        "20",
        "--jax.platform",
        "cpu",
        "--logger.outputs",
        "jsonl",
        "--logdir",
        str(logdir),
    ]
    env = {"PYTHONPATH": str(vendor_root), **dict()}
    completed = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
