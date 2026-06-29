from facet_probe.providers import provider_env_status


def test_provider_env_status_redacts_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")

    status = provider_env_status("openai")

    assert status["ok"] is True
    assert status["required_env"] == {"OPENAI_API_KEY": True}
    assert "not-a-real-key" not in str(status)


def test_provider_env_status_reports_missing_required_env(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    status = provider_env_status("google")

    assert status["ok"] is False
    assert status["required_env"] == {"GOOGLE_API_KEY": False}
