# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""WorldFlux upstream-parity harness utilities."""

from __future__ import annotations

from .backend_adapters import (
    BackendAdapter,
    BackendAdapterRegistry,
    DreamerOfficialJAXSubprocessAdapter,
    DreamerWorldFluxJAXSubprocessAdapter,
    NativeTorchReferenceAdapter,
    TDMPC2OfficialTorchSubprocessAdapter,
    get_backend_adapter_registry,
)
from .backend_contract import (
    ArtifactManifest,
    BackendRunSpec,
    discover_artifacts,
    stable_recipe_hash,
)
from .badge import generate_badge_svg, save_badge
from .campaign import (
    AggregatedResult,
    CampaignRunOptions,
    CampaignSpec,
    MultiSeedCampaign,
    ParityReport,
    export_campaign_source,
    load_campaign_spec,
    parse_seed_csv,
    render_campaign_summary,
    run_campaign,
)
from .harness import aggregate_runs, render_markdown_report, run_suite
from .paper_baselines import (
    DREAMERV3_ATARI100K_BASELINES,
    SUITE_BASELINES,
    TDMPC2_DMCONTROL39_BASELINES,
    PaperBaseline,
)
from .paper_comparison import PaperComparisonReport, PaperDelta, compare_against_paper
from .stats import (
    AggregatedScores,
    EffectSizeResult,
    FDRCorrectionResult,
    MannWhitneyResult,
    WelchTTestResult,
    aggregate_scores,
    benjamini_hochberg,
    cohens_d,
    mann_whitney_u_test,
    welch_t_test,
)

__all__ = [
    "run_suite",
    "aggregate_runs",
    "render_markdown_report",
    "load_campaign_spec",
    "parse_seed_csv",
    "run_campaign",
    "export_campaign_source",
    "render_campaign_summary",
    "CampaignSpec",
    "CampaignRunOptions",
    "PaperBaseline",
    "DREAMERV3_ATARI100K_BASELINES",
    "TDMPC2_DMCONTROL39_BASELINES",
    "SUITE_BASELINES",
    "PaperDelta",
    "PaperComparisonReport",
    "compare_against_paper",
    "generate_badge_svg",
    "save_badge",
    "ArtifactManifest",
    "BackendAdapter",
    "BackendAdapterRegistry",
    "BackendRunSpec",
    "DreamerOfficialJAXSubprocessAdapter",
    "DreamerWorldFluxJAXSubprocessAdapter",
    "NativeTorchReferenceAdapter",
    "TDMPC2OfficialTorchSubprocessAdapter",
    "discover_artifacts",
    "get_backend_adapter_registry",
    "stable_recipe_hash",
    # ML-02: Multi-seed campaign
    "MultiSeedCampaign",
    "AggregatedResult",
    "ParityReport",
    # ML-05: Extended statistics
    "welch_t_test",
    "mann_whitney_u_test",
    "cohens_d",
    "benjamini_hochberg",
    "aggregate_scores",
    "WelchTTestResult",
    "MannWhitneyResult",
    "EffectSizeResult",
    "FDRCorrectionResult",
    "AggregatedScores",
]
