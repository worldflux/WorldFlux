"""Tests for weight mapping between official and WorldFlux implementations."""

from __future__ import annotations

import torch

from worldflux.parity.weight_map import (
    WeightMap,
    WeightMapEntry,
    dreamerv3_weight_map,
    tdmpc2_weight_map,
)


class TestWeightMapEntry:
    def test_basic_mapping(self) -> None:
        entry = WeightMapEntry("a/w", "b.weight")
        assert entry.official_key == "a/w"
        assert entry.worldflux_key == "b.weight"
        assert entry.transpose is False
        assert entry.reshape is None

    def test_transpose_flag(self) -> None:
        entry = WeightMapEntry("a/w", "b.weight", transpose=True)
        assert entry.transpose is True

    def test_reshape_option(self) -> None:
        entry = WeightMapEntry("a/w", "b.weight", reshape=(4, 8))
        assert entry.reshape == (4, 8)


class TestWeightMap:
    def test_official_to_worldflux_basic(self) -> None:
        entries = (
            WeightMapEntry("layer/w", "layer.weight"),
            WeightMapEntry("layer/b", "layer.bias"),
        )
        wm = WeightMap(family="test", entries=entries)
        official = {
            "layer/w": torch.ones(3, 4),
            "layer/b": torch.zeros(4),
        }
        result = wm.official_to_worldflux(official)
        assert "layer.weight" in result
        assert "layer.bias" in result
        assert result["layer.weight"].shape == (3, 4)

    def test_official_to_worldflux_transpose(self) -> None:
        entries = (WeightMapEntry("w", "weight", transpose=True),)
        wm = WeightMap(family="test", entries=entries)
        official = {"w": torch.randn(3, 5)}
        result = wm.official_to_worldflux(official)
        assert result["weight"].shape == (5, 3)

    def test_official_to_worldflux_reshape(self) -> None:
        entries = (WeightMapEntry("w", "weight", reshape=(2, 6)),)
        wm = WeightMap(family="test", entries=entries)
        official = {"w": torch.randn(12)}
        result = wm.official_to_worldflux(official)
        assert result["weight"].shape == (2, 6)

    def test_missing_keys_skipped(self) -> None:
        entries = (
            WeightMapEntry("exists", "a.weight"),
            WeightMapEntry("missing", "b.weight"),
        )
        wm = WeightMap(family="test", entries=entries)
        official = {"exists": torch.ones(2)}
        result = wm.official_to_worldflux(official)
        assert "a.weight" in result
        assert "b.weight" not in result

    def test_validate_coverage(self) -> None:
        entries = (
            WeightMapEntry("a", "x"),
            WeightMapEntry("b", "y"),
        )
        wm = WeightMap(family="test", entries=entries)
        unmapped_off, unmapped_wf = wm.validate_coverage({"a", "b", "c"}, {"x", "y", "z"})
        assert unmapped_off == {"c"}
        assert unmapped_wf == {"z"}

    def test_validate_coverage_full_match(self) -> None:
        entries = (WeightMapEntry("a", "x"),)
        wm = WeightMap(family="test", entries=entries)
        unmapped_off, unmapped_wf = wm.validate_coverage({"a"}, {"x"})
        assert unmapped_off == set()
        assert unmapped_wf == set()


class TestDreamerV3WeightMap:
    def test_creates_valid_map(self) -> None:
        wm = dreamerv3_weight_map()
        assert wm.family == "dreamerv3"
        assert len(wm.entries) > 0

    def test_has_rssm_entries(self) -> None:
        wm = dreamerv3_weight_map()
        wf_keys = {e.worldflux_key for e in wm.entries}
        assert "rssm.gru.weight_ih" in wf_keys
        assert "rssm.prior_net.0.weight" in wf_keys
        assert "rssm.posterior_net.0.weight" in wf_keys

    def test_has_head_entries(self) -> None:
        wm = dreamerv3_weight_map()
        wf_keys = {e.worldflux_key for e in wm.entries}
        assert "reward_head.mlp.0.weight" in wf_keys
        assert "continue_head.mlp.0.weight" in wf_keys

    def test_linear_weights_transposed(self) -> None:
        wm = dreamerv3_weight_map()
        for entry in wm.entries:
            if entry.worldflux_key.endswith(".weight") and "layer_norm" not in entry.official_key:
                assert (
                    entry.transpose is True
                ), f"Linear weight {entry.worldflux_key} should be transposed (JAX->PyTorch)"

    def test_layernorm_not_transposed(self) -> None:
        wm = dreamerv3_weight_map()
        for entry in wm.entries:
            if "layer_norm" in entry.official_key or "scale" in entry.official_key:
                assert entry.transpose is False


class TestTDMPC2WeightMap:
    def test_creates_valid_map(self) -> None:
        wm = tdmpc2_weight_map()
        assert wm.family == "tdmpc2"
        assert len(wm.entries) > 0

    def test_has_component_entries(self) -> None:
        wm = tdmpc2_weight_map()
        wf_keys = {e.worldflux_key for e in wm.entries}
        assert "encoder.0.weight" in wf_keys
        assert "dynamics.0.weight" in wf_keys
        assert "reward_head.0.weight" in wf_keys
        assert "policy.0.weight" in wf_keys

    def test_has_q_ensemble_entries(self) -> None:
        wm = tdmpc2_weight_map()
        wf_keys = {e.worldflux_key for e in wm.entries}
        assert "q_networks.0.0.weight" in wf_keys
        assert "q_networks.4.0.weight" in wf_keys

    def test_no_transpose(self) -> None:
        wm = tdmpc2_weight_map()
        for entry in wm.entries:
            assert entry.transpose is False
