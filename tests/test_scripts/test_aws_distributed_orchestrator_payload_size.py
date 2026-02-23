"""Tests that SSM command payloads stay within the 10 KB delivery limit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "parity"
        / "aws_distributed_orchestrator.py"
    )
    spec = importlib.util.spec_from_file_location("aws_distributed_orchestrator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load aws_distributed_orchestrator")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SSM_PARAMETER_LIMIT = 10_240  # bytes


def test_payload_size_within_ssm_limit() -> None:
    """Full-sized shard commands must fit in the SSM 10 KB parameter limit."""
    mod = _load_module()

    shard = mod.ShardPlan(
        shard_id=39,
        instance_id="i-004d14103e183ef15",
        task_ids=("dog-run",),
        shard_run_id="parity_proof_20260221T013335Z_pilot_shard39",
        estimated_cost=1.0,
        estimated_duration_sec=1.0,
        seed_values=(9,),
        systems=("worldflux",),
        gpu_slot=7,
    )

    commands = mod._build_remote_commands(
        shard=shard,
        manifest_rel="scripts/parity/manifests/official_vs_worldflux_v1.yaml",
        run_id="parity_proof_20260221T013335Z_pilot",
        s3_prefix="s3://worldflux-parity/parity_proof_20260221T013335Z/pilot",
        device="cuda",
        max_retries=2,
        workspace_root="/opt/parity",
        bootstrap_root="/opt/parity/bootstrap",
        worldflux_sha="",
        dreamer_sha="b65cf81a6fb13625af8722127459283f899a35d9",
        tdmpc2_sha="8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe",
        sync_interval_sec=300,
        resume_from_s3=True,
        thread_limit_profile="strict1",
        cpu_affinity_policy="p4d_8gpu_12vcpu",
        skip_bootstrap=False,
    )

    payload = json.dumps(commands)
    assert (
        len(payload) < SSM_PARAMETER_LIMIT
    ), f"JSON payload {len(payload)} bytes exceeds SSM limit {SSM_PARAMETER_LIMIT}"


def test_progress_update_code_is_valid_python() -> None:
    """The helper script must be syntactically valid Python."""
    mod = _load_module()
    code = mod._progress_update_code()
    compile(code, "<progress_update>", "exec")
    assert "os.replace(" in code


def test_write_progress_helper_produces_base64_command() -> None:
    """The write-helper command must decode to the same source."""
    import base64

    mod = _load_module()
    cmd = mod._write_progress_helper_command()
    assert cmd.startswith("echo ")
    assert "base64 -d" in cmd

    b64_part = cmd.split("echo ", 1)[1].split(" | ", 1)[0]
    decoded = base64.b64decode(b64_part).decode()
    assert decoded == mod._progress_update_code()


def test_upload_artifact_skips_zero_byte_json(monkeypatch, tmp_path: Path) -> None:
    mod = _load_module()
    calls: list[list[str]] = []

    def fake_run_cli(command: list[str]):
        calls.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(mod, "_run_cli", fake_run_cli)

    empty_json = tmp_path / "empty.json"
    empty_json.write_text("", encoding="utf-8")
    mod._upload_artifact(region="us-west-2", artifact=empty_json, final_prefix="s3://bucket/final")
    assert not calls

    nonempty_json = tmp_path / "ok.json"
    nonempty_json.write_text("{}", encoding="utf-8")
    mod._upload_artifact(
        region="us-west-2",
        artifact=nonempty_json,
        final_prefix="s3://bucket/final",
    )
    assert calls
