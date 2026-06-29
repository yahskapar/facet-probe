import json
import os
import subprocess
import sys
from pathlib import Path

from facet_probe.reports import build_evaluation_report, write_evaluation_report

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["files"]["summary_json"] == "summary.json"


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
    assert quiet.returncode == 0, quiet.stderr
    assert quiet.stderr == ""
