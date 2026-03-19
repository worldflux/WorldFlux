# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for parity dashboard rendering script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "render_parity_dashboard.py"
    spec = importlib.util.spec_from_file_location("render_parity_dashboard", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load render_parity_dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_dashboard_html_contains_suite_rows(tmp_path: Path) -> None:
    aggregate_path = tmp_path / "aggregate.json"
    output_path = tmp_path / "dashboard.html"
    aggregate_path.write_text(
        """
{
  "generated_at_utc": "2026-03-18T00:00:00Z",
  "all_suites_pass": false,
  "suite_pass_count": 1,
  "suite_fail_count": 1,
  "suites": [
    {
      "suite_id": "dreamer_suite",
      "family": "dreamerv3",
      "pass_non_inferiority": true,
      "ci_upper_ratio": 0.01,
      "margin_ratio": 0.05,
      "verdict_reason": "PASS"
    },
    {
      "suite_id": "tdmpc2_suite",
      "family": "tdmpc2",
      "pass_non_inferiority": false,
      "ci_upper_ratio": 0.12,
      "margin_ratio": 0.05,
      "verdict_reason": "FAIL"
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    mod = _load_script_module()
    mod.render_dashboard(aggregate_path, output_path=output_path)

    html = output_path.read_text(encoding="utf-8")
    assert "dreamer_suite" in html
    assert "tdmpc2_suite" in html
    assert "WorldFlux Parity Dashboard" in html
