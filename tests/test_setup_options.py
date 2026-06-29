import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_setup(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "setup.sh", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_environment_yml_defines_conda_environment():
    env = yaml.safe_load((REPO_ROOT / "environment.yml").read_text(encoding="utf-8"))

    assert env["name"] == "facet-probe"
    assert "conda-forge" in env["channels"]
    assert any(str(dep).startswith("python=3.11") for dep in env["dependencies"])
    assert any(str(dep).startswith("pip") for dep in env["dependencies"])


def test_setup_script_is_executable_and_documents_both_modes():
    script = REPO_ROOT / "setup.sh"
    help_result = run_setup("--help")

    assert os.access(script, os.X_OK)
    assert help_result.returncode == 0
    assert "bash setup.sh conda" in help_result.stdout
    assert "bash setup.sh uv" in help_result.stdout
    assert "--accelerators MODE" in help_result.stdout


def test_setup_conda_dry_run_does_not_require_conda():
    result = run_setup("conda", "--dry-run", "--prefix", "/tmp/facet-probe-test-env")

    assert result.returncode == 0
    assert (
        "conda env create --prefix /tmp/facet-probe-test-env --file environment.yml"
        in result.stdout
    )
    normalized = result.stdout.replace("\\", "")
    assert ".[dev,hf,analysis,models,providers]" in normalized
    assert ".[accelerators]" in normalized


def test_setup_uv_dry_run_does_not_require_uv():
    result = run_setup("uv", "--dry-run", "--extras", "dev,hf,analysis")
    normalized = result.stdout.replace("\\", "")

    assert result.returncode == 0
    assert "uv venv --python 3.11 .venv" in result.stdout
    assert "uv pip install --python" in result.stdout
    assert ".[dev,hf,analysis]" in normalized
    assert ".[accelerators]" in normalized


def test_setup_dry_run_can_skip_or_require_accelerators():
    skip = run_setup("uv", "--dry-run", "--accelerators", "no")
    require = run_setup("uv", "--dry-run", "--accelerators", "yes")

    assert skip.returncode == 0
    assert ".[accelerators]" not in skip.stdout.replace("\\", "")
    assert require.returncode == 0
    assert ".[accelerators]" in require.stdout.replace("\\", "")


def test_setup_dry_run_supports_irt_extra():
    uv_result = run_setup(
        "uv",
        "--dry-run",
        "--extras",
        "dev,hf,analysis,models,providers,irt",
    )
    conda_result = run_setup(
        "conda",
        "--dry-run",
        "--prefix",
        "/tmp/facet-probe-irt-test-env",
        "--extras",
        "dev,hf,analysis,irt",
    )

    assert uv_result.returncode == 0
    assert ".[dev,hf,analysis,models,providers,irt]" in uv_result.stdout.replace("\\", "")
    assert conda_result.returncode == 0
    assert ".[dev,hf,analysis,irt]" in conda_result.stdout.replace("\\", "")
