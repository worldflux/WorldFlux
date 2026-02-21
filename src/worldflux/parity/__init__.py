"""WorldFlux upstream-parity harness utilities."""

from .campaign import (
    CampaignRunOptions,
    CampaignSpec,
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
]
