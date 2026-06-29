"""Evaluation report artifact writers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from facet_probe.figures import write_report_figures
from facet_probe.metrics import (
    audit_records,
    item_metrics_as_dicts,
    summarize_groups,
    write_csv,
    write_json,
)


def build_evaluation_report(
    records: list[dict[str, Any]],
    *,
    label: str,
    group_by: tuple[str, ...] = ("facet", "dataset", "model"),
    include_items: bool = True,
) -> dict[str, Any]:
    summary = asdict(audit_records(records, label=label))
    groups = summarize_groups(records, group_by=group_by)
    report: dict[str, Any] = {
        "schema_version": 1,
        "label": label,
        "group_by": list(group_by),
        "summary": summary,
        "groups": groups,
    }
    if include_items:
        report["items"] = item_metrics_as_dicts(records)
    return report


def write_evaluation_report(
    output_dir: str | Path,
    records: list[dict[str, Any]],
    *,
    label: str,
    group_by: tuple[str, ...] = ("facet", "dataset", "model"),
    include_items: bool = True,
    include_figures: bool = True,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = build_evaluation_report(
        records,
        label=label,
        group_by=group_by,
        include_items=include_items,
    )
    paths = {
        "summary_json": output / "summary.json",
        "group_csv": output / "group_summary.csv",
        "manifest_json": output / "report_manifest.json",
    }
    write_json(paths["summary_json"], report["summary"])
    write_csv(paths["group_csv"], report["groups"])
    if include_items:
        paths["item_csv"] = output / "item_metrics.csv"
        write_csv(paths["item_csv"], report["items"])
    if include_figures:
        figure_paths = write_report_figures(report, output / "figures")
        paths.update({f"figures_{name}": path for name, path in figure_paths.items()})

    manifest = {
        "schema_version": report["schema_version"],
        "label": label,
        "group_by": list(group_by),
        "files": {
            name: str(path.relative_to(output))
            for name, path in paths.items()
            if name != "manifest_json"
        },
    }
    write_json(paths["manifest_json"], manifest)
    return paths
