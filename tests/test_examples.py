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
