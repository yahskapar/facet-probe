import json
import os
import subprocess
import sys
from pathlib import Path

import facet_probe as fp
from facet_probe.judging import judge_mixed_trials, parse_judge_response

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parse_judge_response_extracts_verdict_and_reason():
    verdict, reason = parse_judge_response(
        "Verdict: B\nReason: One ordering gives a different city."
    )

    assert verdict == "B"
    assert reason == "One ordering gives a different city."


def test_judge_mixed_trials_mock_writes_semantic_summary(tmp_path):
    records = [
        _mixed_record("mramg", "model-a", "item-1", 0, "flour"),
        _mixed_record("mramg", "model-a", "item-1", 1, "flour"),
        _mixed_record("mramg", "model-a", "item-2", 0, "flour"),
        _mixed_record("mramg", "model-a", "item-2", 1, "sugar"),
    ]

    status = judge_mixed_trials(records, output_dir=tmp_path, mock_judge=True)

    assert status["status"] == "completed"
    assert status["summary"][0]["sem_flip"] == 0.5
    assert status["summary"][0]["text_flip_upper"] == 0.5
    assert (tmp_path / "mixed_semantic_judgments.jsonl").exists()
    assert (tmp_path / "mixed_semantic_summary.csv").exists()


def test_judge_mixed_cli_mock(tmp_path):
    trials = tmp_path / "trials.jsonl"
    trials.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _mixed_record("mmqa", "model-a", "item-1", 0, "1995"),
                _mixed_record("mmqa", "model-a", "item-1", 1, "1996"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "judge"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "facet_probe.cli",
            "judge-mixed",
            str(trials),
            "--mock-judge",
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["summary"][0]["sem_flip"] == 1.0
    assert (output / "mixed_semantic_summary.json").exists()


def test_judge_mixed_cli_default_judge_reports_missing_api_key(tmp_path):
    trials = tmp_path / "trials.jsonl"
    trials.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                _mixed_record("mmqa", "model-a", "item-1", 0, "1995"),
                _mixed_record("mmqa", "model-a", "item-1", 1, "1996"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    env.pop("GOOGLE_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "facet_probe.cli",
            "judge-mixed",
            str(trials),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 2
    assert "mixed-semantic-primary" in result.stderr
    assert "GOOGLE_API_KEY" in result.stderr


def test_package_exports_judging_api():
    assert fp.parse_judge_response("Verdict: A")[0] == "A"
    assert fp.judge_mixed_trials is judge_mixed_trials


def _mixed_record(dataset, model, item_id, ordering_idx, answer):
    return {
        "facet": "mixed_modality_order",
        "dataset": dataset,
        "model": model,
        "item_id": item_id,
        "ordering_idx": ordering_idx,
        "answer_normalized": answer,
        "raw_output": answer,
        "question": "What is the answer?",
    }
