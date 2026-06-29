"""Publication-oriented figure writers for reports and ODI/IRT summaries."""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any

FIGURE_FORMATS = ("png", "pdf")
MAX_GROUP_ROWS = 12
MAX_TOP_ITEM_ROWS = 15
MAX_THETA_ROWS = 24
MAX_FACET_ROWS = 16
MIN_MEANINGFUL_OSI = 1e-12
FACET_PROBE_BLUE = "#3563a9"
FACET_PROBE_TEAL = "#2b8a7e"
FACET_PROBE_RED = "#b4443f"
FACET_PROBE_GOLD = "#b9832f"
FACET_PROBE_GRAY = "#5f6773"
GRID_COLOR = "#d7dce2"


def write_report_figures(
    report: dict[str, Any],
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = FIGURE_FORMATS,
) -> dict[str, Path]:
    """Write default run-level figures from a built evaluation report."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "figure_type": "evaluation_report",
        "figures": {},
    }

    try:
        plt = _load_pyplot()
    except Exception as exc:  # pragma: no cover - depends on optional matplotlib extra
        manifest["status"] = "skipped"
        manifest["reason"] = str(exc)
        manifest_path = output / "figures_manifest.json"
        _write_json(manifest_path, manifest)
        return {"figures_manifest_json": manifest_path}

    summary = report.get("summary") or {}
    groups = list(report.get("groups") or [])
    items = list(report.get("items") or [])
    _apply_style(plt)

    summary_paths = _plot_summary_overview(plt, summary, output, formats=formats)
    _add_paths(paths, "figure_summary_overview", summary_paths)
    manifest["figures"]["summary_overview"] = _rel_paths(output, summary_paths)

    if groups:
        group_paths = _plot_group_metrics(
            plt,
            groups,
            tuple(report.get("group_by") or ()),
            output,
            formats=formats,
        )
        _add_paths(paths, "figure_group_metrics", group_paths)
        manifest["figures"]["group_metrics"] = _rel_paths(output, group_paths)

    if items:
        dist_paths = _plot_item_instability_distribution(
            plt,
            items,
            output,
            formats=formats,
        )
        top_paths = _plot_top_unstable_items(plt, items, output, formats=formats)
        _add_paths(paths, "figure_item_instability", dist_paths)
        _add_paths(paths, "figure_top_unstable_items", top_paths)
        manifest["figures"]["item_instability_distribution"] = _rel_paths(output, dist_paths)
        manifest["figures"]["top_unstable_items"] = _rel_paths(output, top_paths)

    manifest_path = output / "figures_manifest.json"
    _write_json(manifest_path, manifest)
    paths["figures_manifest_json"] = manifest_path
    return paths


def write_irt_outcome_figures(
    output_dir: str | Path,
    *,
    outcome: str,
    theta_rows: list[dict[str, Any]] | None = None,
    facet_rows: list[dict[str, Any]] | None = None,
    title_prefix: str = "ODI/IRT",
    formats: tuple[str, ...] = FIGURE_FORMATS,
) -> dict[str, Path]:
    """Write theta and facet-decomposition figures for one ODI/IRT outcome."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "figure_type": "irt_outcome",
        "outcome": outcome,
        "figures": {},
    }

    try:
        plt = _load_pyplot()
    except Exception as exc:  # pragma: no cover - depends on optional matplotlib extra
        manifest["status"] = "skipped"
        manifest["reason"] = str(exc)
        manifest_path = output / f"irt_{outcome}_figures_manifest.json"
        _write_json(manifest_path, manifest)
        return {f"{outcome}_figures_manifest_json": manifest_path}

    _apply_style(plt)
    if theta_rows:
        theta_paths = _plot_theta_intervals(
            plt,
            theta_rows,
            output,
            outcome=outcome,
            title_prefix=title_prefix,
            formats=formats,
        )
        _add_paths(paths, f"{outcome}_figure_theta", theta_paths)
        manifest["figures"]["theta_intervals"] = _rel_paths(output, theta_paths)
    if facet_rows:
        facet_paths = _plot_facet_decomposition(
            plt,
            facet_rows,
            output,
            outcome=outcome,
            title_prefix=title_prefix,
            formats=formats,
        )
        _add_paths(paths, f"{outcome}_figure_facet_decomposition", facet_paths)
        manifest["figures"]["facet_decomposition"] = _rel_paths(output, facet_paths)

    manifest_path = output / f"irt_{outcome}_figures_manifest.json"
    _write_json(manifest_path, manifest)
    paths[f"{outcome}_figures_manifest_json"] = manifest_path
    return paths


def write_released_irt_figures(
    summary: dict[str, Any],
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = FIGURE_FORMATS,
) -> dict[str, Path]:
    """Write paper-artifact figures from the released ODI/IRT summary."""

    paths: dict[str, Path] = {}
    for outcome, payload in (summary.get("outcomes") or {}).items():
        outcome_paths = write_irt_outcome_figures(
            output_dir,
            outcome=outcome,
            theta_rows=list((payload.get("theta") or {}).get("models") or []),
            facet_rows=list(summary.get("facet_decomposition") or [])
            if outcome == "modal"
            else None,
            title_prefix="Released ODI/IRT",
            formats=formats,
        )
        paths.update({f"released_{name}": path for name, path in outcome_paths.items()})
    return paths


def _load_pyplot():
    cache_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/facet-probe-matplotlib"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def _apply_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _plot_summary_overview(
    plt: Any,
    summary: dict[str, Any],
    output: Path,
    *,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    metrics = _summary_metrics(summary)
    labels = [item[0] for item in metrics]
    values = [float(item[1]) for item in metrics]
    colors = [item[2] for item in metrics]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    bars = ax.bar(labels, values, color=colors, width=0.62)
    max_value = max(values) if values else 1.0
    if max_value <= 1.0:
        ax.set_ylim(0, 1)
        ax.set_ylabel("Rate")
    else:
        ax.set_ylim(0, max_value * 1.18)
        ax.set_ylabel("Count")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ax.get_ylim()[1] * 0.025,
            _format_metric(value),
            ha="center",
            va="bottom",
        )
    ax.set_title(f"{summary.get('label', 'run')} summary")
    ax.set_xlabel("")
    ax.grid(axis="x", visible=False)
    return _save_figure(fig, output, "summary_overview", formats=formats)


def _plot_group_metrics(
    plt: Any,
    groups: list[dict[str, Any]],
    group_by: tuple[str, ...],
    output: Path,
    *,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    metric_defs = _group_metric_defs(groups)
    sorted_groups = sorted(
        groups,
        key=_group_sort_key,
        reverse=True,
    )
    n_total = len(sorted_groups)
    sorted_groups = sorted_groups[:MAX_GROUP_ROWS]
    labels = _deduplicate_labels([_group_label(row, group_by) for row in sorted_groups])
    height = max(3.4, 0.38 * len(labels) + 1.2)
    fig, axes = plt.subplots(
        1,
        len(metric_defs),
        figsize=(4.2 * len(metric_defs), height),
        sharey=True,
    )
    if len(metric_defs) == 1:
        axes = [axes]
    y_positions = list(range(len(labels)))
    for ax, (key, title, color) in zip(axes, metric_defs, strict=True):
        values = [_float_or_none(row.get(key)) or 0.0 for row in sorted_groups]
        ax.barh(y_positions, values, color=color, alpha=0.92)
        ax.set_title(title)
        ax.set_xlim(0, 1 if max(values or [0]) <= 1 else max(values) * 1.12)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        for y_pos, value in zip(y_positions, values, strict=True):
            ax.text(
                min(value + 0.015, ax.get_xlim()[1] * 0.98),
                y_pos,
                _format_metric(value),
                va="center",
            )
    axes[0].set_yticks(y_positions, labels)
    title = "Grouped ordering-sensitivity metrics"
    if n_total > len(sorted_groups):
        title += f" (top {len(sorted_groups)} of {n_total})"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return _save_figure(fig, output, "group_metrics", formats=formats)


def _plot_item_instability_distribution(
    plt: Any,
    items: list[dict[str, Any]],
    output: Path,
    *,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    osi = [_float_or_none(row.get("osi")) for row in items]
    modal_fraction = [_float_or_none(row.get("modal_fraction")) for row in items]
    osi_values = [_normalize_unit_value(float(value)) for value in osi if value is not None]
    modal_values = [float(value) for value in modal_fraction if value is not None]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    axes[0].hist(
        osi_values,
        bins=_bins_for_unit_interval(osi_values),
        color=FACET_PROBE_BLUE,
        alpha=0.88,
    )
    axes[0].set_title("Item OSI distribution")
    axes[0].set_xlabel("OSI")
    axes[0].set_ylabel("Items")
    axes[0].set_xlim(0, 1)
    axes[1].hist(
        modal_values,
        bins=_bins_for_unit_interval(modal_values),
        color=FACET_PROBE_GOLD,
        alpha=0.88,
    )
    axes[1].set_title("Modal-answer concentration")
    axes[1].set_xlabel("Modal fraction")
    axes[1].set_ylabel("Items")
    axes[1].set_xlim(0, 1)
    fig.tight_layout()
    return _save_figure(fig, output, "item_instability_distribution", formats=formats)


def _plot_top_unstable_items(
    plt: Any,
    items: list[dict[str, Any]],
    output: Path,
    *,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    unstable = _select_top_unstable_items(items, limit=MAX_TOP_ITEM_ROWS)
    n_unstable = sum(1 for row in items if _positive_osi(row) > MIN_MEANINGFUL_OSI)
    if not unstable:
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        _draw_empty_state(ax, "No item-level OSI above zero")
        ax.set_title("Most order-sensitive items")
        return _save_figure(fig, output, "top_unstable_items", formats=formats)

    ranked = unstable
    labels = _deduplicate_labels([_item_label(row) for row in ranked])
    values = [_positive_osi(row) for row in ranked]
    colors = [FACET_PROBE_RED if row.get("flipped") else FACET_PROBE_GRAY for row in ranked]
    height = max(3.4, 0.36 * len(labels) + 1.2)
    fig, ax = plt.subplots(figsize=(8.2, height))
    y_positions = list(range(len(labels)))
    ax.barh(y_positions, values, color=colors, alpha=0.9)
    ax.set_yticks(y_positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("OSI")
    title = "Most order-sensitive items"
    if n_unstable > len(ranked):
        title += f" (top {len(ranked)} of {n_unstable})"
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    for y_pos, value in zip(y_positions, values, strict=True):
        ax.text(min(value + 0.015, 0.98), y_pos, _format_metric(value), va="center")
    fig.tight_layout()
    return _save_figure(fig, output, "top_unstable_items", formats=formats)


def _plot_theta_intervals(
    plt: Any,
    theta_rows: list[dict[str, Any]],
    output: Path,
    *,
    outcome: str,
    title_prefix: str,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    all_rows = sorted(theta_rows, key=lambda row: _float_or_none(row.get("theta_mean")) or 0.0)
    rows = _select_theta_rows(all_rows, limit=MAX_THETA_ROWS)
    labels = [
        _wrap_label(
            _compact_component(str(row.get("model") or row.get("display") or "model"), 36),
            width=28,
        )
        for row in rows
    ]
    labels = _deduplicate_labels(labels)
    means = [_float_or_none(row.get("theta_mean")) or 0.0 for row in rows]
    lows = [_float_or_none(row.get("theta_2.5")) for row in rows]
    highs = [_float_or_none(row.get("theta_97.5")) for row in rows]
    height = max(3.4, 0.34 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(7.6, height))
    y_positions = list(range(len(rows)))
    for y_pos, mean, low, high in zip(y_positions, means, lows, highs, strict=True):
        if low is not None and high is not None:
            ax.plot([low, high], [y_pos, y_pos], color=FACET_PROBE_GRAY, linewidth=1.4)
        ax.scatter(mean, y_pos, color=FACET_PROBE_BLUE, s=28, zorder=3)
    ax.axvline(0, color="#222222", linewidth=0.8, alpha=0.75)
    ax.set_yticks(y_positions, labels)
    ax.set_xlabel("Model theta posterior mean with 95% interval")
    title = f"{title_prefix} {outcome} model theta"
    if len(all_rows) > len(rows):
        title += f" (lowest/highest {len(rows)} of {len(all_rows)})"
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save_figure(fig, output, f"irt_{outcome}_theta_intervals", formats=formats)


def _plot_facet_decomposition(
    plt: Any,
    facet_rows: list[dict[str, Any]],
    output: Path,
    *,
    outcome: str,
    title_prefix: str,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    all_rows = sorted(
        facet_rows,
        key=lambda row: _float_or_none(row.get("sigma_pi_mean")) or 0.0,
        reverse=True,
    )
    rows = all_rows[:MAX_FACET_ROWS]
    labels = _deduplicate_labels(
        [
            _wrap_label(_compact_component(str(row.get("facet", "facet")), 30), width=24)
            for row in rows
        ]
    )
    sigma = [_float_or_none(row.get("sigma_pi_mean")) or 0.0 for row in rows]
    delta_abs = [_float_or_none(row.get("delta_abs_mean")) for row in rows]
    delta_values = [0.0 if value is None else float(value) for value in delta_abs]
    y_positions = list(range(len(rows)))
    height = max(3.4, 0.42 * len(rows) + 1.3)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, height), sharey=True)
    axes[0].barh(y_positions, sigma, color=FACET_PROBE_BLUE, alpha=0.9)
    axes[0].set_title("Facet order sensitivity")
    axes[0].set_xlabel("sigma_pi mean")
    axes[0].set_yticks(y_positions, labels)
    axes[0].invert_yaxis()
    axes[0].grid(axis="y", visible=False)
    _plot_delta_effect_axis(axes[1], y_positions, delta_values)
    axes[1].set_title("Mean absolute directional effect")
    axes[1].set_xlabel("|delta| mean")
    axes[1].grid(axis="y", visible=False)
    title = f"{title_prefix} {outcome} facet decomposition"
    if len(all_rows) > len(rows):
        title += f" (top {len(rows)} of {len(all_rows)})"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    return _save_figure(fig, output, f"irt_{outcome}_facet_decomposition", formats=formats)


def _plot_delta_effect_axis(ax: Any, y_positions: list[int], values: list[float]) -> None:
    positive = [value for value in values if value > 0]
    if not positive:
        ax.barh(y_positions, values, color=FACET_PROBE_RED, alpha=0.88)
        return
    use_log = max(positive) / min(positive) > 20
    if not use_log:
        ax.barh(y_positions, values, color=FACET_PROBE_RED, alpha=0.88)
        return
    x_min = min(positive) * 0.55
    x_max = max(positive) * 1.35
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    for y_pos, value in zip(y_positions, values, strict=True):
        if value <= 0:
            continue
        ax.hlines(y_pos, x_min, value, color=FACET_PROBE_RED, linewidth=1.8, alpha=0.55)
        ax.scatter(value, y_pos, color=FACET_PROBE_RED, s=42, zorder=3)
        ax.text(value * 1.08, y_pos, _format_metric(value), va="center", fontsize=8)
    ax.text(
        0.98,
        0.03,
        "log scale",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=FACET_PROBE_GRAY,
        fontsize=8,
    )


def _save_figure(
    fig: Any,
    output: Path,
    stem: str,
    *,
    formats: tuple[str, ...],
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for fmt in formats:
        path = output / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        paths[fmt] = path
    fig.clf()
    try:
        from matplotlib import pyplot as plt

        plt.close(fig)
    except Exception:
        pass
    return paths


def _add_paths(target: dict[str, Path], prefix: str, paths: dict[str, Path]) -> None:
    for fmt, path in paths.items():
        target[f"{prefix}_{fmt}"] = path


def _rel_paths(base: Path, paths: dict[str, Path]) -> dict[str, str]:
    return {fmt: str(path.relative_to(base)) for fmt, path in paths.items()}


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary_metrics(summary: dict[str, Any]) -> list[tuple[str, float, str]]:
    metrics = [
        ("Flip rate", _float_or_none(summary.get("flip_rate")), FACET_PROBE_RED),
        ("Mean OSI", _float_or_none(summary.get("mean_osi")), FACET_PROBE_BLUE),
        ("Macro accuracy", _float_or_none(summary.get("macro_accuracy")), FACET_PROBE_TEAL),
    ]
    present = [(label, float(value), color) for label, value, color in metrics if value is not None]
    if present:
        return present

    for label, key in (
        ("Parseable trials", "n_parseable_trials"),
        ("Trials", "n_trials"),
        ("Items", "n_items"),
    ):
        value = _float_or_none(summary.get(key))
        if value is not None:
            return [(label, float(value), FACET_PROBE_GRAY)]
    return [("Items", 0.0, FACET_PROBE_GRAY)]


def _group_metric_defs(groups: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    rate_defs = [
        ("flip_rate", "Flip rate", FACET_PROBE_RED),
        ("mean_osi", "Mean OSI", FACET_PROBE_BLUE),
        ("macro_accuracy", "Macro accuracy", FACET_PROBE_TEAL),
    ]
    present_rate_defs = [item for item in rate_defs if _any_group_metric(groups, item[0])]
    if present_rate_defs:
        return present_rate_defs

    count_defs = [
        ("n_items", "Items", FACET_PROBE_BLUE),
        ("n_trials", "Trials", FACET_PROBE_TEAL),
        ("n_parseable_trials", "Parseable trials", FACET_PROBE_GOLD),
    ]
    present_count_defs = [item for item in count_defs if _any_group_metric(groups, item[0])]
    return present_count_defs or [("n_items", "Items", FACET_PROBE_GRAY)]


def _group_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    flip_rate = _normalize_unit_value(_float_or_none(row.get("flip_rate")) or 0.0)
    mean_osi = _normalize_unit_value(_float_or_none(row.get("mean_osi")) or 0.0)
    n_items = _float_or_none(row.get("n_items")) or 0.0
    n_trials = _float_or_none(row.get("n_trials")) or 0.0
    return (max(flip_rate, mean_osi), mean_osi, flip_rate, n_items, n_trials)


def _select_top_unstable_items(
    items: list[dict[str, Any]],
    *,
    limit: int = MAX_TOP_ITEM_ROWS,
) -> list[dict[str, Any]]:
    unstable = [row for row in items if _positive_osi(row) > MIN_MEANINGFUL_OSI]
    return sorted(
        unstable,
        key=lambda row: (
            _positive_osi(row),
            int(row.get("n_distinct_answers") or 0),
            int(row.get("n_parseable") or 0),
        ),
        reverse=True,
    )[:limit]


def _select_theta_rows(
    rows: list[dict[str, Any]],
    *,
    limit: int = MAX_THETA_ROWS,
) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    lower = limit // 2
    upper = limit - lower
    return rows[:lower] + rows[-upper:]


def _positive_osi(row: dict[str, Any]) -> float:
    return _normalize_unit_value(_float_or_none(row.get("osi")) or 0.0)


def _normalize_unit_value(value: float) -> float:
    if abs(value) <= MIN_MEANINGFUL_OSI:
        return 0.0
    return max(0.0, min(1.0, value))


def _draw_empty_state(ax: Any, message: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=FACET_PROBE_GRAY,
        fontsize=10,
    )


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _any_group_metric(groups: list[dict[str, Any]], key: str) -> bool:
    return any(_float_or_none(row.get(key)) is not None for row in groups)


def _format_metric(value: float) -> str:
    if abs(value) <= MIN_MEANINGFUL_OSI:
        return "0"
    if abs(value) < 0.001:
        return f"{value:.1e}"
    if abs(value) < 0.01:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _group_label(row: dict[str, Any], group_by: tuple[str, ...]) -> str:
    fields = group_by or tuple(key for key in ("facet", "dataset", "model") if key in row)
    parts = [
        _compact_component(
            str(row.get(field)),
            _component_max_chars(field),
        )
        for field in fields
        if row.get(field) is not None
    ]
    return _wrap_label(" / ".join(parts), width=34)


def _item_label(row: dict[str, Any]) -> str:
    raw_parts = [
        _compact_component(str(row.get("dataset") or ""), 24),
        _compact_component(str(row.get("model") or ""), 24),
        _compact_component(str(row.get("item_id") or ""), 30),
    ]
    parts = [part for part in raw_parts if part]
    text = " / ".join(parts)
    return _wrap_label(text, width=42)


def _component_max_chars(field: str) -> int:
    return {
        "facet": 22,
        "dataset": 20,
        "model": 20,
        "item_id": 30,
    }.get(field, 24)


def _compact_component(value: str, max_chars: int) -> str:
    text = _facet_label(_collapse_repeated_suffix(value.strip()))
    return _shorten_middle(text, max_chars)


def _collapse_repeated_suffix(value: str, *, max_run: int = 8) -> str:
    if len(value) <= max_run:
        return value
    last = value[-1]
    run_length = 0
    for char in reversed(value):
        if char != last:
            break
        run_length += 1
    if run_length <= max_run:
        return value
    return value[:-run_length] + last * 3


def _shorten_middle(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    keep = max_chars - 3
    head = (keep + 1) // 2
    tail = keep - head
    return f"{value[:head]}...{value[-tail:]}"


def _deduplicate_labels(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        count = seen.get(label, 0) + 1
        seen[label] = count
        out.append(label if count == 1 else f"{label}\n#{count}")
    return out


def _facet_label(value: str) -> str:
    return value.replace("_", " ")


def _wrap_label(value: str, *, width: int) -> str:
    wrapped = textwrap.wrap(
        value,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    return "\n".join(wrapped) or value


def _bins_for_unit_interval(values: list[float]) -> list[float] | int:
    if not values:
        return 10
    return [idx / 10 for idx in range(11)]
