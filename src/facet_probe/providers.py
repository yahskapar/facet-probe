"""Provider metadata for safe reproduction environment checks."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    required_env: tuple[str, ...]
    optional_env: tuple[str, ...] = ()
    notes: str = ""


PROVIDERS: dict[str, ProviderSpec] = {
    "google": ProviderSpec(
        name="google",
        required_env=("GOOGLE_API_KEY",),
        optional_env=("FACET_PROBE_GOOGLE_TIMEOUT_S", "FACET_PROBE_GOOGLE_MAX_RETRIES"),
        notes="Gemini-family closed-source models.",
    ),
    "openai": ProviderSpec(
        name="openai",
        required_env=("OPENAI_API_KEY",),
        optional_env=("FACET_PROBE_OPENAI_TIMEOUT_S", "FACET_PROBE_OPENAI_MAX_RETRIES"),
        notes="OpenAI/ChatGPT-family closed-source models.",
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        required_env=("ANTHROPIC_API_KEY",),
        optional_env=("FACET_PROBE_ANTHROPIC_TIMEOUT_S", "FACET_PROBE_ANTHROPIC_MAX_RETRIES"),
        notes="Claude-family closed-source models.",
    ),
    "openai-compatible": ProviderSpec(
        name="openai-compatible",
        required_env=("FACET_PROBE_OPENAI_COMPAT_BASE_URL",),
        optional_env=("FACET_PROBE_OPENAI_COMPAT_API_KEY",),
        notes="Local or hosted OpenAI-compatible endpoints such as vLLM.",
    ),
    "huggingface": ProviderSpec(
        name="huggingface",
        required_env=(),
        optional_env=("HF_TOKEN", "HF_HOME"),
        notes="Open-weight local inference or gated HuggingFace downloads.",
    ),
    "mock": ProviderSpec(
        name="mock",
        required_env=(),
        notes="Deterministic local adapter for smoke tests and artifact validation.",
    ),
}


def get_provider(name: str) -> ProviderSpec:
    try:
        return PROVIDERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown provider {name!r}; known providers: {sorted(PROVIDERS)}") from exc


def provider_env_status(name: str) -> dict[str, object]:
    spec = get_provider(name)
    required = {key: bool(os.environ.get(key)) for key in spec.required_env}
    optional = {key: bool(os.environ.get(key)) for key in spec.optional_env}
    return {
        "provider": spec.name,
        "ok": all(required.values()) if required else True,
        "required_env": required,
        "optional_env": optional,
        "notes": spec.notes,
    }
