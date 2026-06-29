import json

from facet_probe.reports import build_evaluation_report, write_evaluation_report

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
