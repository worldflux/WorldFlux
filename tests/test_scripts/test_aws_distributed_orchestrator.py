"""Tests for AWS distributed parity orchestrator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
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


def test_aws_distributed_orchestrator_wait_mode_with_mocked_aws(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_module()

    manifest = {
        "schema_version": "parity.manifest.v1",
        "defaults": {"alpha": 0.05, "equivalence_margin": 0.05},
        "seed_policy": {
            "mode": "fixed",
            "values": [0],
            "pilot_seeds": 10,
            "min_seeds": 20,
            "max_seeds": 50,
            "power_target": 0.8,
        },
        "tasks": [
            {
                "task_id": "atari100k_pong",
                "family": "dreamerv3",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_dreamerv3",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')"],
                },
                "worldflux": {
                    "adapter": "worldflux_dreamerv3_native",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')"],
                },
            },
            {
                "task_id": "dog-run",
                "family": "tdmpc2",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_tdmpc2",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')"],
                },
                "worldflux": {
                    "adapter": "worldflux_tdmpc2_native",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')"],
                },
            },
        ],
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def fake_run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["aws", "ssm", "send-command"]:
            instance = command[command.index("--instance-ids") + 1]
            return subprocess.CompletedProcess(command, 0, stdout=f"cmd-{instance}\n", stderr="")

        if command[:3] == ["aws", "ssm", "get-command-invocation"]:
            payload = {
                "Status": "Success",
                "ResponseCode": 0,
                "StatusDetails": "Success",
                "StandardOutputContent": "ok",
                "StandardErrorContent": "",
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

        if command[:3] == ["aws", "s3", "cp"]:
            src = command[3]
            dst = command[4]
            if src.startswith("s3://"):
                target = Path(dst)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.name == "parity_runs.jsonl":
                    shard = target.parent.name.split("_")[-1]
                    task_id = "atari100k_pong" if shard == "00" else "dog-run"
                    rows = [
                        {
                            "schema_version": "parity.v1",
                            "task_id": task_id,
                            "seed": 0,
                            "system": "official",
                            "status": "success",
                            "metrics": {"final_return_mean": 100.0, "auc_return": 90.0},
                        },
                        {
                            "schema_version": "parity.v1",
                            "task_id": task_id,
                            "seed": 0,
                            "system": "worldflux",
                            "status": "success",
                            "metrics": {"final_return_mean": 99.0, "auc_return": 89.0},
                        },
                    ]
                    target.write_text(
                        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
                    )
                elif target.name == "seed_plan.json":
                    target.write_text(json.dumps({"seed_values": [0]}), encoding="utf-8")
                elif target.name == "run_context.json":
                    target.write_text(json.dumps({"seeds": [0]}), encoding="utf-8")
                elif target.name == "run_summary.json":
                    target.write_text(json.dumps({"ok": True}), encoding="utf-8")
                elif target.name == "coverage_report.json":
                    target.write_text(
                        json.dumps({"missing_pairs": 0, "pass": True}), encoding="utf-8"
                    )
            else:
                if not Path(src).exists():
                    return subprocess.CompletedProcess(
                        command, 1, stdout="", stderr="missing upload source"
                    )
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def fake_run_local_script(script: Path, args: list[str]) -> None:
        if script.name == "merge_parity_runs.py":
            output = Path(args[args.index("--output") + 1])
            summary = Path(args[args.index("--summary-output") + 1])
            inputs = [
                Path(args[idx + 1]) for idx, token in enumerate(args[:-1]) if token == "--input"
            ]
            rows: list[dict] = []
            for path in inputs:
                rows.extend(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            summary.write_text(json.dumps({"merged_records": len(rows)}), encoding="utf-8")
            return

        if script.name == "validate_matrix_completeness.py":
            out = Path(args[args.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"missing_pairs": 0, "pass": True}), encoding="utf-8")
            return

        if script.name == "stats_equivalence.py":
            out = Path(args[args.index("--output") + 1])
            validity_out = None
            if "--validity-report" in args:
                validity_out = Path(args[args.index("--validity-report") + 1])
            report = {
                "schema_version": "parity.v1",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "input": "merged",
                "global": {
                    "tasks_total": 2,
                    "tasks_pass_primary": 2,
                    "tasks_pass_all_metrics": 2,
                    "parity_pass_primary": True,
                    "parity_pass_all_metrics": True,
                    "parity_pass_final": True,
                    "missing_pairs": 0,
                    "strict_mode_failed": False,
                },
                "tasks": [],
                "completeness": {
                    "expected_pairs": 4,
                    "missing_pairs": 0,
                    "task_seed_count": 2,
                    "complete_task_seed_pairs": 2,
                },
                "validity": {
                    "proof_mode": True,
                    "pass": True,
                    "issue_count": 0,
                    "issues": [],
                },
                "holm": {"primary": {}, "all_metrics": {}},
                "config": {},
            }
            out.write_text(json.dumps(report), encoding="utf-8")
            if validity_out is not None:
                validity_out.write_text(
                    json.dumps(
                        {
                            "schema_version": "parity.v1",
                            "proof_mode": True,
                            "pass": True,
                            "issue_count": 0,
                            "issues": [],
                        }
                    ),
                    encoding="utf-8",
                )
            return

        if script.name == "report_markdown.py":
            out = Path(args[args.index("--output") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("# report\n", encoding="utf-8")
            return

        raise AssertionError(f"unexpected script: {script}")

    monkeypatch.setattr(mod, "_run_cli", fake_run_cli)
    monkeypatch.setattr(mod, "_run_local_script", fake_run_local_script)

    output_dir = tmp_path / "out"
    argv = [
        "aws_distributed_orchestrator.py",
        "--region",
        "us-west-2",
        "--instance-ids",
        "i-aaaa,i-bbbb",
        "--manifest",
        str(manifest_path),
        "--run-id",
        "parity_test",
        "--s3-prefix",
        "s3://bucket/parity_test",
        "--output-dir",
        str(output_dir),
    ]

    monkeypatch.setattr(sys, "argv", argv)
    rc = mod.main()
    assert rc == 0

    summary_path = output_dir / "parity_test" / "orchestrator_summary.json"
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["failed_shards"] == 0
    assert Path(summary["artifacts"]["equivalence_report"]).exists()
    assert Path(summary["artifacts"]["coverage_report"]).exists()
