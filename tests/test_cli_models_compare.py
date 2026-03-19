# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for worldflux models compare CLI command."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from worldflux import cli

runner = CliRunner()


class TestModelsCompare:
    def test_compare_two_models(self):
        result = runner.invoke(cli.app, ["models", "compare", "dreamerv3:size12m", "tdmpc2:5m"])
        assert result.exit_code == 0
        assert "dreamerv3:size12m" in result.output
        assert "tdmpc2:5m" in result.output

    def test_compare_same_family(self):
        result = runner.invoke(
            cli.app, ["models", "compare", "dreamerv3:size12m", "dreamerv3:size50m"]
        )
        assert result.exit_code == 0

    def test_compare_json(self):
        result = runner.invoke(
            cli.app, ["models", "compare", "dreamerv3:size12m", "tdmpc2:5m", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "model_1" in data
        assert "model_2" in data

    def test_compare_unknown_model(self):
        result = runner.invoke(
            cli.app, ["models", "compare", "dreamerv3:size12m", "nonexistent:xxx"]
        )
        assert result.exit_code != 0
