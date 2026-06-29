import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_python_library_usage_example_runs():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    result = subprocess.run(
        [sys.executable, "examples/python_library_usage.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["facet_probe_version"] == "0.0.1"
    assert payload["validation"]["ok"] is True
    assert payload["manifest_rows"] == 3
    assert payload["summary"]["n_trials"] == 3


def test_quickstart_profile_example_runs():
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    env.pop("GOOGLE_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "examples/quickstart_profile.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["facet_probe_version"] == "0.0.1"
    assert payload["paper_profile"]["k_orderings"] == 6
    assert payload["paper_profile"]["seed"] == 42
    assert payload["closed_source_model"]["provider"] == "google"
    assert payload["open_weight_model"]["provider"] == "huggingface"
    assert payload["mixed_semantic_judge"]["name"] == "mixed-semantic-primary"
    assert payload["judge_env_status"]["required_env"] == {"GOOGLE_API_KEY": False}
    assert sorted(payload["provider_env_status"]) == ["gemini-3.1-pro-preview", "qwen3-5-4b"]
    assert payload["custom_only_datasets"] == ["arc_challenge"]
    assert payload["manifest_rows_for_demo_item"] == 6
