import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "facet_probe.cli", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=run_env,
    )


def test_paper_run_accepts_arbitrary_hf_model(tmp_path):
    result = run_cli(
        "paper-run",
        "--hf-model",
        "Qwen/Qwen3.5-VL-4B-Instruct",
        "--output-dir",
        str(tmp_path / "qwen-paper"),
        "--prepare-only",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "prepared"
    assert payload["profile"]["models"][0]["provider"] == "huggingface"
    assert (tmp_path / "qwen-paper" / "run_profile.json").exists()
    assert (tmp_path / "qwen-paper" / "models.jsonl").exists()


def test_paper_run_accepts_closed_source_model_and_api_key(tmp_path):
    result = run_cli(
        "paper-run",
        "--provider",
        "google",
        "--api-model",
        "gemini-3.1-pro-preview",
        "--api-key-env",
        "GOOGLE_API_KEY",
        "--output-dir",
        str(tmp_path / "gemini-paper"),
        "--prepare-only",
        env={"GOOGLE_API_KEY": "not-a-real-key"},
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["provider_status"]["gemini-3.1-pro-preview"]["ok"] is True
    assert payload["profile"]["models"][0]["provider"] == "google"


def test_paper_run_uses_model_config_and_overrides_k_seed(tmp_path):
    result = run_cli(
        "paper-run",
        "--model-config",
        "configs/models.yaml",
        "--models",
        "gemini-3.1-pro-preview",
        "qwen3-5-4b",
        "--k",
        "3",
        "--seed",
        "7",
        "--output-dir",
        str(tmp_path / "paper-config"),
        "--prepare-only",
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["profile"]["k_orderings"] == 3
    assert payload["profile"]["seed"] == 7
    assert [model["name"] for model in payload["profile"]["models"]] == [
        "gemini-3.1-pro-preview",
        "qwen3-5-4b",
    ]


def test_paper_run_executes_mock_model_and_writes_results(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text(
        json.dumps(
            {
                "item_id": "toy::1",
                "dataset": "toy",
                "question_ref": "Which option is blue?",
                "components": [
                    {
                        "component_id": "choice_0",
                        "kind": "choice",
                        "content_ref": "toy-choice-a",
                        "label": "A",
                    },
                    {
                        "component_id": "choice_1",
                        "kind": "choice",
                        "content_ref": "toy-choice-b",
                        "label": "B",
                    },
                ],
                "choices": ["red", "blue"],
                "gold": "1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "mock-paper"

    result = run_cli(
        "paper-run",
        "--mock-model",
        "deterministic-mock",
        "--items-jsonl",
        str(items),
        "--k",
        "3",
        "--output-dir",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["summary"]["n_trials"] == 3
    assert (output / "manifest.jsonl").exists()
    assert (output / "trials.jsonl").exists()
    assert (output / "summary.json").exists()
    assert (output / "group_summary.csv").exists()
    assert (output / "report" / "item_metrics.csv").exists()

    trial_rows = [
        json.loads(line)
        for line in (output / "trials.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["model"] for row in trial_rows] == ["deterministic-mock"] * 3
    assert {row["correct"] for row in trial_rows} == {True}
