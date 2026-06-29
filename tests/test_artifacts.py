from facet_probe.artifacts import verify_release_artifacts


def test_release_artifacts_verify():
    checks = verify_release_artifacts()

    assert checks
    assert all(check.ok for check in checks), [check for check in checks if not check.ok]
