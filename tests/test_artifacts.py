from facet_probe.artifacts import verify_release_artifacts


def test_release_artifacts_verify():
    checks = verify_release_artifacts()

    assert checks
    assert all(check.ok for check in checks), [check for check in checks if not check.ok]


def test_release_artifacts_load_from_outside_repo_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    checks = verify_release_artifacts()

    assert checks
    assert all(check.ok for check in checks), [check for check in checks if not check.ok]
