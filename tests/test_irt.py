import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from facet_probe.irt import (
    load_irt_input_rows,
    released_irt_summary,
    trial_records_to_irt_rows,
    write_irt_fit,
    write_irt_input,
    write_released_irt_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    return subprocess.run(
        [sys.executable, "-m", "facet_probe.cli", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def test_released_irt_summary_loads_paper_odi_artifacts():
    summary = released_irt_summary()

    assert summary["release"] == "v0.0.1"
    assert summary["paper_usage"]["main_table_2"] == "modal outcome facet decomposition"
    assert len(summary["facet_decomposition"]) == 5
    assert summary["outcomes"]["modal"]["diagnostics"]["diverging"] == 0
    assert summary["outcomes"]["correct"]["diagnostics"]["n_chains"] == 4
    assert len(summary["outcomes"]["correct"]["theta"]["models"]) == 18


def test_write_released_irt_summary_bundle(tmp_path):
    status = write_released_irt_summary(tmp_path / "released_irt")

    assert status["status"] == "completed"
    files = status["files"]
    assert Path(files["summary_json"]).exists()
    assert Path(files["theta_modal_csv"]).exists()
    assert Path(files["theta_correct_csv"]).exists()
    assert Path(files["copied_artifact_dir"], "irt_v4_modal_per_item_params.parquet").exists()


def test_trial_records_to_irt_rows_modal_and_correct():
    records = [
        _trial("i1", 0, "A", True),
        _trial("i1", 1, "A", True),
        _trial("i1", 2, "B", False),
        _trial("i2", 0, "C", True),
        _trial("i2", 1, "D", False),
        _trial("i3", 0, None, None),
    ]

    modal_rows = trial_records_to_irt_rows(records, outcomes=("modal",))
    assert [row["outcome_value"] for row in modal_rows] == [1, 1, 0]
    assert {row["item_id"] for row in modal_rows} == {"i1"}

    correct_rows = trial_records_to_irt_rows(records, outcomes=("correct",))
    assert [row["outcome_value"] for row in correct_rows] == [1, 1, 0, 1, 0]


def test_write_irt_input_outputs_summary_and_groups(tmp_path):
    status = write_irt_input(
        [_trial("i1", 0, "A", True), _trial("i1", 1, "B", False)],
        tmp_path / "irt_input",
    )

    assert status["summary"]["outcomes"]["modal"]["skipped_tied_modal_trials"] == 2
    assert status["summary"]["outcomes"]["correct"]["n_rows"] == 2
    assert Path(status["files"]["trials_csv"]).exists()
    assert Path(status["files"]["trials_jsonl"]).exists()
    rows = list(csv.DictReader(Path(status["files"]["groups_csv"]).open(encoding="utf-8")))
    assert rows[0]["outcome"] == "correct"
    assert rows[0]["mean_outcome"] == "0.5"


def test_write_irt_fit_dry_run_prepares_exported_rows(tmp_path):
    export = write_irt_input(
        [
            _trial("i1", 0, "A", True, model="m1"),
            _trial("i1", 1, "A", True, model="m1"),
            _trial("i1", 0, "B", False, model="m2"),
            _trial("i1", 1, "B", False, model="m2"),
        ],
        tmp_path / "irt_input",
    )

    loaded_csv = load_irt_input_rows(export["files"]["trials_csv"])
    loaded_jsonl = load_irt_input_rows(export["files"]["trials_jsonl"])
    assert len(loaded_csv) == len(loaded_jsonl) == 8

    status = write_irt_fit(
        export["files"]["trials_csv"],
        tmp_path / "irt_fit",
        outcome="modal",
        dry_run=True,
    )

    assert status["status"] == "prepared"
    assert status["fits"][0]["input_summary"]["n_rows"] == 4
    assert status["fits"][0]["input_summary"]["n_models"] == 2
    assert Path(status["files"]["modal_input_summary_json"]).exists()
    assert Path(status["files"]["fit_summary_json"]).exists()


def test_irt_cli_commands(tmp_path):
    summary_result = run_cli("irt-summary", "--output-dir", str(tmp_path / "released_irt"))
    assert summary_result.returncode == 0, summary_result.stderr
    assert (tmp_path / "released_irt" / "released_irt_summary.json").exists()

    trials = tmp_path / "trials.jsonl"
    trials.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _trial("i1", 0, "A", True),
                _trial("i1", 1, "A", True),
                _trial("i1", 2, "B", False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    export_result = run_cli(
        "irt-export",
        str(trials),
        "--output-dir",
        str(tmp_path / "irt_input"),
    )
    assert export_result.returncode == 0, export_result.stderr
    payload = json.loads(export_result.stdout)
    assert payload["summary"]["outcomes"]["modal"]["n_rows"] == 3
    irt_csv = tmp_path / "irt_input" / "irt_input_trials.csv"
    assert irt_csv.exists()

    fit_result = run_cli(
        "irt-fit",
        str(irt_csv),
        "--output-dir",
        str(tmp_path / "irt_fit"),
        "--dry-run",
    )
    assert fit_result.returncode == 0, fit_result.stderr
    fit_payload = json.loads(fit_result.stdout)
    assert fit_payload["status"] == "prepared"
    assert fit_payload["fits"][0]["outcome"] == "modal"
    assert (tmp_path / "irt_fit" / "irt_fit_summary.json").exists()


def _trial(
    item_id: str,
    ordering_idx: int,
    answer: str | None,
    correct: bool | None,
    *,
    model: str = "m",
) -> dict:
    return {
        "facet": "option_order",
        "dataset": "toy",
        "model": model,
        "item_id": item_id,
        "ordering_idx": ordering_idx,
        "permutation": [ordering_idx, 0],
        "answer_normalized": answer,
        "gold_normalized": "A",
        "correct": correct,
    }
