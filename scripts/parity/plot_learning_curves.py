#!/usr/bin/env python3
"""Plot learning curves from parity JSONL results.

Generates per-task comparison plots and a suite-level grid overview.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the parity package is importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from worldflux.parity.curves import aggregate_curves, load_curves_from_parity_jsonl  # noqa: E402

# ── colour scheme ──────────────────────────────────────────────────────
_COLORS: dict[str, str] = {}
_CI_ALPHA = 0.2


def _system_color(system: str) -> str:
    """Return a colour for the given system name."""
    lower = system.lower()
    if "official" in lower:
        return "#1f77b4"  # tab:blue
    return "#ff7f0e"  # tab:orange


def _system_label(system: str) -> str:
    """Human-friendly label for a system identifier."""
    lower = system.lower()
    if "official" in lower:
        return "official"
    if "worldflux" in lower:
        return "worldflux"
    return system


# ── static (matplotlib) plots ──────────────────────────────────────────
def plot_per_task(
    aggregated: dict[tuple[str, str], dict],
    *,
    output_dir: Path,
    fmt: str = "png",
) -> list[Path]:
    """Create one figure per task, each showing systems with CI bands."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks: dict[str, list[tuple[str, str]]] = {}
    for task, system in aggregated:
        tasks.setdefault(task, []).append((task, system))

    saved: list[Path] = []
    for task, keys in sorted(tasks.items()):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in sorted(keys, key=lambda k: k[1]):
            data = aggregated[key]
            system = key[1]
            color = _system_color(system)
            label = _system_label(system)
            ax.plot(data["steps"], data["mean"], color=color, label=label)
            ax.fill_between(
                data["steps"],
                data["ci_low"],
                data["ci_high"],
                color=color,
                alpha=_CI_ALPHA,
            )
        ax.set_title(task)
        ax.set_xlabel("Step")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = output_dir / f"{task}.{fmt}"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        saved.append(out_path)
    return saved


def plot_suite_grid(
    aggregated: dict[tuple[str, str], dict],
    *,
    output_dir: Path,
    fmt: str = "png",
) -> Path | None:
    """Create a grid overview figure with all tasks in one image."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted({k[0] for k in aggregated})
    if not tasks:
        return None

    ncols = min(3, len(tasks))
    nrows = math.ceil(len(tasks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for idx, task in enumerate(tasks):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        keys = sorted(
            [k for k in aggregated if k[0] == task],
            key=lambda k: k[1],
        )
        for key in keys:
            data = aggregated[key]
            system = key[1]
            color = _system_color(system)
            label = _system_label(system)
            ax.plot(data["steps"], data["mean"], color=color, label=label)
            ax.fill_between(
                data["steps"],
                data["ci_low"],
                data["ci_high"],
                color=color,
                alpha=_CI_ALPHA,
            )
        ax.set_title(task, fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots.
    for idx in range(len(tasks), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    out_path = output_dir / f"suite_overview.{fmt}"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return out_path


# ── interactive (plotly) plots ─────────────────────────────────────────
def plot_interactive(
    aggregated: dict[tuple[str, str], dict],
    *,
    output_dir: Path,
) -> Path | None:
    """Generate an interactive plotly HTML overview (optional dependency)."""
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
        from plotly.subplots import make_subplots  # type: ignore[import-untyped]
    except ImportError:
        return None

    tasks = sorted({k[0] for k in aggregated})
    if not tasks:
        return None

    fig = make_subplots(
        rows=len(tasks),
        cols=1,
        subplot_titles=tasks,
        vertical_spacing=0.05,
    )

    for row_idx, task in enumerate(tasks, start=1):
        keys = sorted([k for k in aggregated if k[0] == task], key=lambda k: k[1])
        for key in keys:
            data = aggregated[key]
            system = key[1]
            color = _system_color(system)
            label = _system_label(system)
            fig.add_trace(
                go.Scatter(
                    x=data["steps"].tolist(),
                    y=data["mean"].tolist(),
                    name=label,
                    line=dict(color=color),
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=data["steps"].tolist(),
                    y=data["ci_high"].tolist(),
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=data["steps"].tolist(),
                    y=data["ci_low"].tolist(),
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=color.replace(")", ",0.2)").replace("rgb", "rgba")
                    if color.startswith("rgb")
                    else color + "33",
                    showlegend=False,
                ),
                row=row_idx,
                col=1,
            )

    fig.update_layout(height=350 * len(tasks), title_text="Parity Learning Curves")
    out_path = output_dir / "suite_interactive.html"
    fig.write_html(str(out_path))
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot learning curves from parity JSONL results.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a merged JSONL file or directory of JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/parity/plots"),
        help="Directory for output plots.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("png", "pdf", "svg"),
        default="png",
        dest="fmt",
        help="Output image format (default: png).",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Also generate plotly interactive HTML (requires plotly).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    curves = load_curves_from_parity_jsonl(args.input)
    if not curves:
        print("No curves found in input.", file=sys.stderr)
        sys.exit(1)

    aggregated = aggregate_curves(curves)
    if not aggregated:
        print("No aggregatable curve data (need >=2 points per seed).", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_task = plot_per_task(aggregated, output_dir=args.output_dir, fmt=args.fmt)
    for p in per_task:
        print(f"Saved: {p}")

    grid = plot_suite_grid(aggregated, output_dir=args.output_dir, fmt=args.fmt)
    if grid:
        print(f"Saved: {grid}")

    if args.interactive:
        interactive = plot_interactive(aggregated, output_dir=args.output_dir)
        if interactive:
            print(f"Saved: {interactive}")
        else:
            print("plotly not installed; skipping interactive output.", file=sys.stderr)


if __name__ == "__main__":
    main()
