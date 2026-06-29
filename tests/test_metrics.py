import json

from facet_probe.metrics import audit_records, item_metrics, summarize_groups, write_csv, write_json
from facet_probe.schema import TrialRecord


def test_item_metrics_and_summary_detect_any_flip():
    records = [
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
        {
            "facet": "option_order",
            "dataset": "toy",
            "model": "m",
            "item_id": "i2",
            "ordering_idx": 0,
            "answer_normalized": "0",
            "correct": True,
        },
        {
            "facet": "option_order",
            "dataset": "toy",
            "model": "m",
            "item_id": "i2",
            "ordering_idx": 1,
            "answer_normalized": "0",
            "correct": True,
        },
    ]

    per_item = item_metrics(records)
    summary = audit_records(records, label="toy")
    groups = summarize_groups(records)

    assert [m.flipped for m in per_item] == [True, False]
    assert summary.flip_rate == 0.5
    assert summary.macro_accuracy == 0.75
    assert groups[0]["flip_rate"] == 0.5
    assert groups[0]["n_trials"] == 4


def test_trial_record_coerces_string_correct_values():
    true_row = TrialRecord.from_mapping({"item_id": "i1", "correct": "true"})
    false_row = TrialRecord.from_mapping({"item_id": "i2", "correct": "false"})
    unknown_row = TrialRecord.from_mapping({"item_id": "i3", "correct": "not sure"})

    assert true_row.correct is True
    assert false_row.correct is False
    assert unknown_row.correct is None


def test_output_helpers_create_parent_directories(tmp_path):
    json_path = tmp_path / "nested" / "summary.json"
    csv_path = tmp_path / "nested" / "summary.csv"

    write_json(json_path, {"ok": True})
    write_csv(csv_path, [{"facet": "option_order", "flip_rate": 0.5}])

    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "facet,flip_rate"
