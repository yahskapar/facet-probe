import json
import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

from facet_probe.figures import (
    MAX_THETA_ROWS,
    _format_metric,
    _group_label,
    _group_metric_defs,
    _item_label,
    _select_theta_rows,
    _select_top_unstable_items,
)
from facet_probe.reports import build_evaluation_report, write_evaluation_report

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_MATPLOTLIB = find_spec("matplotlib") is not None

RECORDS = [
    {
        "facet": "option_order",
        "dataset": "toy",
        "model": "m",
        "item_id": "i1",
        "ordering_idx": 0,
        "answer_normalized": "0",
        "correct": True,
    },
    {
        "facet": "option_order",
        "dataset": "toy",
        "model": "m",
        "item_id": "i1",
        "ordering_idx": 1,
        "answer_normalized": "1",
        "correct": False,
    },
]


def test_build_evaluation_report_includes_summary_groups_and_items():
    report = build_evaluation_report(RECORDS, label="toy-run")

    assert report["summary"]["label"] == "toy-run"
    assert report["summary"]["flip_rate"] == 1.0
    assert report["groups"][0]["macro_accuracy"] == 0.5
    assert report["items"][0]["n_distinct_answers"] == 2


def test_write_evaluation_report_creates_artifact_bundle(tmp_path):
    paths = write_evaluation_report(tmp_path / "report", RECORDS, label="toy-run")

    assert paths["summary_json"].exists()
    assert paths["group_csv"].exists()
    assert paths["item_csv"].exists()
    assert paths["figures_figures_manifest_json"].exists()
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["files"]["summary_json"] == "summary.json"
    assert manifest["files"]["figures_figures_manifest_json"] == "figures/figures_manifest.json"
    figure_manifest = json.loads(
        paths["figures_figures_manifest_json"].read_text(encoding="utf-8")
    )
    if HAS_MATPLOTLIB:
        assert paths["figures_figure_summary_overview_png"].exists()
        assert paths["figures_figure_group_metrics_pdf"].exists()
        assert paths["figures_figure_top_unstable_items_png"].exists()
        assert paths["figures_figure_summary_overview_png"].read_bytes().startswith(
            b"\x89PNG"
        )
        assert (
            manifest["files"]["figures_figure_summary_overview_png"]
            == "figures/summary_overview.png"
        )
        assert figure_manifest["status"] == "completed"
    else:
        assert figure_manifest["status"] == "skipped"


def test_figure_helpers_filter_stable_items_and_compact_long_labels():
    long_id = "commonsenseqa::" + "abcdef1234567890" * 4
    items = [
        {
            "dataset": "commonsenseqa",
            "model": "qwen3-5-4b",
            "item_id": "stable",
            "osi": -0.0,
            "n_distinct_answers": 1,
        },
        {
            "dataset": "commonsenseqa",
            "model": "qwen3-5-4b",
            "item_id": long_id,
            "osi": 0.65,
            "n_distinct_answers": 2,
        },
        {
            "dataset": "mmlu_pro",
            "model": "qwen3-5-4b",
            "item_id": "zero",
            "osi": 0.0,
            "n_distinct_answers": 1,
        },
    ]

    selected = _select_top_unstable_items(items)

    assert [row["item_id"] for row in selected] == [long_id]
    assert _select_top_unstable_items([items[0], items[2]]) == []
    label = _item_label(items[1])
    assert "..." in label
    assert long_id not in label
    assert all(len(line) <= 42 for line in label.splitlines())


def test_figure_helpers_fallback_counts_and_cap_dense_theta_rows():
    count_defs = _group_metric_defs([{"dataset": "toy", "n_items": 4, "n_trials": 12}])
    rate_defs = _group_metric_defs([{"dataset": "toy", "flip_rate": 0.2, "n_items": 4}])
    label = _group_label(
        {
            "facet": "evidence_chunk_order",
            "dataset": "dataset_" + "x" * 80,
            "model": "model_" + "y" * 80,
        },
        ("facet", "dataset", "model"),
    )
    rows = [{"model": f"m{i}", "theta_mean": float(i)} for i in range(MAX_THETA_ROWS + 10)]
    selected = _select_theta_rows(rows)

    assert [item[0] for item in count_defs] == ["n_items", "n_trials"]
    assert [item[0] for item in rate_defs] == ["flip_rate"]
    assert "x" * 20 not in label
    assert "y" * 20 not in label
    assert all(len(line) <= 34 for line in label.splitlines())
    assert len(selected) == MAX_THETA_ROWS
    assert selected[0]["model"] == "m0"
    assert selected[-1]["model"] == f"m{MAX_THETA_ROWS + 9}"
    assert _format_metric(0.0001) == "1.0e-04"
    assert _format_metric(-0.0) == "0"


def test_make_report_cli_emits_progress_and_quiet_suppresses_it(tmp_path):
    trials = tmp_path / "trials.jsonl"
    trials.write_text(
        "\n".join(json.dumps(row) for row in RECORDS) + "\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}

    noisy = subprocess.run(
        [
            sys.executable,
            "-m",
            "facet_probe.cli",
            "make-report",
            str(trials),
            "--output-dir",
            str(tmp_path / "report-noisy"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    quiet = subprocess.run(
        [
            sys.executable,
            "-m",
            "facet_probe.cli",
            "make-report",
            str(trials),
            "--output-dir",
            str(tmp_path / "report-quiet"),
            "--quiet",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert noisy.returncode == 0, noisy.stderr
    assert "facet-probe make-report [" in noisy.stderr
    assert "building report over 2 trial record(s)" in noisy.stderr
    payload = json.loads(noisy.stdout)
    assert Path(payload["figures_figures_manifest_json"]).exists()
    if HAS_MATPLOTLIB:
        assert Path(payload["figures_figure_summary_overview_png"]).exists()
    assert quiet.returncode == 0, quiet.stderr
    assert quiet.stderr == ""
