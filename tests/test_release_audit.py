import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_offline_release_audit_passes():
    result = subprocess.run(
        [sys.executable, "scripts/audit_release.py", "--offline-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAIL" not in result.stdout
