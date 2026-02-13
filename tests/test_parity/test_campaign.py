"""Tests for parity campaign orchestration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from worldflux.parity import (
    CampaignRunOptions,
    export_campaign_source,
    load_campaign_spec,
    parse_seed_csv,
    run_campaign,
)
from worldflux.parity.errors import ParityError


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_parse_seed_csv() -> None:
    assert parse_seed_csv("2,1,2,0") == (0, 1, 2)
    assert parse_seed_csv(None) == ()


def test_campaign_exports_from_input_artifact(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    campaign_path = tmp_path / "campaign.yaml"

    _write_json(
        input_path,
        {
            "scores": [
                {"task": "task_a", "seed": 0, "step": 1, "score": 1.0},
                {"task": "task_a", "seed": 0, "step": 3, "score": 1.5},
                {"task": "task_b", "seed": 0, "step": 2, "score": 2.0},
            ]
        },
    )
    _write_json(
        campaign_path,
        {
            "schema_version": "worldflux.parity.campaign.v1",
            "suite_id": "toy_suite",
            "family": "dreamerv3",
            "default_step": 3,
            "default_seeds": [0],
            "tasks": ["task_a", "task_b"],
            "sources": {
                "worldflux": {
                    "input_path": str(input_path),
                    "input_format": "canonical_json",
                    "output_path": str(output_path),
                }
            },
        },
    )

    spec = load_campaign_spec(campaign_path)
    summary = run_campaign(
        spec,
        CampaignRunOptions(
            mode="worldflux",
            seeds=(0,),
            device="cpu",
            output=output_path,
            oracle_output=None,
            resume=False,
            dry_run=False,
            workdir=tmp_path,
            pair_output_root=None,
        ),
    )
    assert summary["suite_id"] == "toy_suite"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    scores = payload["scores"]
    assert len(scores) == 2
    task_a = next(row for row in scores if row["task"] == "task_a")
    assert task_a["step"] == 3
    assert task_a["score"] == 1.5


def test_campaign_runs_command_template(tmp_path: Path) -> None:
    output_path = tmp_path / "worldflux.json"
    campaign_path = tmp_path / "campaign.yaml"
    writer_path = tmp_path / "writer.py"
    writer_path.write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--task', required=True)",
                "parser.add_argument('--seed', type=int, required=True)",
                "parser.add_argument('--output', required=True)",
                "args = parser.parse_args()",
                "path = Path(args.output)",
                "path.parent.mkdir(parents=True, exist_ok=True)",
                "payload = {'scores': [{'task': args.task, 'seed': args.seed, 'step': 10, 'score': 7.0}]}",
                "path.write_text(json.dumps(payload), encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )

    command_template = (
        f"{sys.executable} {writer_path} --task {{task}} --seed {{seed}} --output {{pair_output}}"
    )
    _write_json(
        campaign_path,
        {
            "schema_version": "worldflux.parity.campaign.v1",
            "suite_id": "command_suite",
            "family": "tdmpc2",
            "default_step": 10,
            "default_seeds": [1, 2],
            "tasks": ["task_x"],
            "sources": {
                "worldflux": {
                    "command_template": command_template,
                    "result_format": "canonical_json",
                    "output_path": str(output_path),
                }
            },
        },
    )

    spec = load_campaign_spec(campaign_path)
    run_campaign(
        spec,
        CampaignRunOptions(
            mode="worldflux",
            seeds=(1, 2),
            device="cpu",
            output=output_path,
            oracle_output=None,
            resume=False,
            dry_run=False,
            workdir=tmp_path,
            pair_output_root=tmp_path / "pairs",
        ),
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    keys = {(row["task"], row["seed"]) for row in payload["scores"]}
    assert keys == {("task_x", 1), ("task_x", 2)}


def test_campaign_command_template_with_unbalanced_quote_fails(tmp_path: Path) -> None:
    output_path = tmp_path / "worldflux.json"
    campaign_path = tmp_path / "campaign.yaml"
    command_template = "python -c 'print(1) --task {task}"
    _write_json(
        campaign_path,
        {
            "schema_version": "worldflux.parity.campaign.v1",
            "suite_id": "command_suite_invalid",
            "family": "tdmpc2",
            "default_step": 10,
            "default_seeds": [1],
            "tasks": ["task_x"],
            "sources": {
                "worldflux": {
                    "command_template": command_template,
                    "result_format": "canonical_json",
                    "output_path": str(output_path),
                }
            },
        },
    )

    spec = load_campaign_spec(campaign_path)
    with pytest.raises(ParityError, match="Invalid command_template"):
        run_campaign(
            spec,
            CampaignRunOptions(
                mode="worldflux",
                seeds=(1,),
                device="cpu",
                output=output_path,
                oracle_output=None,
                resume=False,
                dry_run=False,
                workdir=tmp_path,
                pair_output_root=tmp_path / "pairs",
            ),
        )


def test_campaign_export_helper(tmp_path: Path) -> None:
    input_path = tmp_path / "oracle.json"
    _write_json(
        input_path,
        {
            "scores": [
                {"task": "task_z", "seed": 3, "step": 5, "score": 11.0},
            ]
        },
    )
    campaign_path = tmp_path / "campaign.json"
    export_path = tmp_path / "exported.json"
    _write_json(
        campaign_path,
        {
            "schema_version": "worldflux.parity.campaign.v1",
            "suite_id": "export_suite",
            "family": "dreamerv3",
            "default_seeds": [3],
            "tasks": ["task_z"],
            "sources": {
                "oracle": {
                    "input_path": str(input_path),
                    "input_format": "canonical_json",
                    "output_path": str(export_path),
                }
            },
        },
    )
    spec = load_campaign_spec(campaign_path)
    summary = export_campaign_source(
        spec,
        source_name="oracle",
        seeds=(3,),
        output_path=export_path,
        resume=False,
    )
    assert summary["mode"] == "oracle"
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    assert payload["score_count"] == 1
