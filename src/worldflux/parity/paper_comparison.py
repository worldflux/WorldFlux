"""Compare parity run scores against published paper baselines."""

from __future__ import annotations

from dataclasses import dataclass

from .paper_baselines import SUITE_BASELINES, PaperBaseline


@dataclass(frozen=True)
class PaperDelta:
    """Per-task delta between a run score and the paper baseline."""

    task: str
    paper_score: float
    run_score: float
    absolute_delta: float
    relative_delta_pct: float


@dataclass(frozen=True)
class PaperComparisonReport:
    """Aggregate comparison of run scores against paper baselines."""

    suite_id: str
    deltas: tuple[PaperDelta, ...]
    mean_relative_delta_pct: float
    tasks_within_5pct: int

    def render_markdown(self) -> str:
        lines: list[str] = []
        lines.append("## Paper Baseline Comparison")
        lines.append("")
        lines.append(f"Suite: `{self.suite_id}`")
        lines.append("")
        lines.append("| Task | Paper Score | Run Score | Delta | Relative % |")
        lines.append("|---|---:|---:|---:|---:|")
        for delta in self.deltas:
            lines.append(
                f"| {delta.task} | {delta.paper_score:.1f} | {delta.run_score:.1f} "
                f"| {delta.absolute_delta:+.1f} | {delta.relative_delta_pct:+.1f}% |"
            )
        lines.append("")
        lines.append(f"Mean relative delta: **{self.mean_relative_delta_pct:+.1f}%**")
        lines.append(f"Tasks within 5%: **{self.tasks_within_5pct}** / {len(self.deltas)}")
        return "\n".join(lines) + "\n"


def compare_against_paper(
    suite_id: str,
    run_scores: dict[str, float],
) -> PaperComparisonReport | None:
    """Match run scores against paper baselines for the given suite.

    Returns ``None`` when no baseline table is registered for *suite_id*.
    """
    baselines = SUITE_BASELINES.get(suite_id)
    if baselines is None:
        return None

    deltas: list[PaperDelta] = []
    for task_id in sorted(run_scores.keys()):
        baseline: PaperBaseline | None = baselines.get(task_id)
        if baseline is None:
            continue
        run_score = run_scores[task_id]
        absolute_delta = run_score - baseline.score
        relative_delta_pct = absolute_delta / max(abs(baseline.score), 1.0) * 100.0
        deltas.append(
            PaperDelta(
                task=task_id,
                paper_score=baseline.score,
                run_score=run_score,
                absolute_delta=absolute_delta,
                relative_delta_pct=relative_delta_pct,
            )
        )

    if not deltas:
        return None

    mean_rel = sum(d.relative_delta_pct for d in deltas) / len(deltas)
    within_5 = sum(1 for d in deltas if abs(d.relative_delta_pct) <= 5.0)

    return PaperComparisonReport(
        suite_id=suite_id,
        deltas=tuple(deltas),
        mean_relative_delta_pct=mean_rel,
        tasks_within_5pct=within_5,
    )
