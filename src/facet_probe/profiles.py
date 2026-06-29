"""Adapter-neutral evaluation profiles for Facet-Probe runs."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from facet_probe.artifacts import release_data_root
from facet_probe.hf_inspect import normalize_hf_dataset_id, sanitize_dataset_name
from facet_probe.providers import get_provider, provider_env_status

try:
    from importlib.resources.abc import Traversable
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    from importlib.abc import Traversable

_MODEL_METADATA_KEYS = {
    "display",
    "family",
    "cluster",
    "provider",
    "api_model_id",
    "api_key_env",
    "hf_repo",
    "note",
    "role",
}

NameSelection = str | Sequence[str] | None
ConfigPath = Path | Traversable


@dataclass(frozen=True)
class ModelProfile:
    """Common model configuration passed to a closed or open model adapter."""

    name: str
    provider: str
    model_id: str
    adapter: str
    env: tuple[str, ...] = ()
    hf_repo: str | None = None
    generation: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def env_status(self) -> dict[str, object]:
        status = dict(provider_env_status(self.provider))
        if self.env:
            required = {key: bool(os.environ.get(key)) for key in self.env}
            status["required_env"] = required
            status["ok"] = all(required.values())
        return status


@dataclass(frozen=True)
class DatasetProfile:
    """Dataset selection and facet metadata for an evaluation profile."""

    name: str
    hf_repo: str
    split: str
    facets: tuple[str, ...]
    source: str = "huggingface"
    config: str | None = None
    revision: str | None = None
    license: str | None = None
    audited_n: int | str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationProfile:
    """Serializable profile describing what an adapter-backed evaluation runs."""

    name: str
    models: tuple[ModelProfile, ...]
    datasets: tuple[DatasetProfile, ...]
    k_orderings: int = 6
    seed: int = 42
    generation_defaults: dict[str, Any] = field(default_factory=dict)
    config_paths: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def provider_status(self) -> dict[str, dict[str, object]]:
        return {model.name: model.env_status() for model in self.models}

    def with_datasets(
        self,
        *datasets: DatasetProfile,
        name: str | None = None,
        replace_existing: bool = False,
    ) -> EvaluationProfile:
        if replace_existing:
            merged = tuple(datasets)
        else:
            by_name = {dataset.name: dataset for dataset in self.datasets}
            for dataset in datasets:
                by_name[dataset.name] = dataset
            merged = tuple(by_name.values())
        return replace(self, name=name or self.name, datasets=merged)

    def add_datasets(
        self,
        *datasets: DatasetProfile,
        name: str | None = None,
    ) -> EvaluationProfile:
        return self.with_datasets(*datasets, name=name, replace_existing=False)

    def only_datasets(
        self,
        *datasets: DatasetProfile,
        name: str | None = None,
    ) -> EvaluationProfile:
        return self.with_datasets(*datasets, name=name, replace_existing=True)

    def with_models(
        self,
        *models: ModelProfile,
        name: str | None = None,
        replace_existing: bool = False,
    ) -> EvaluationProfile:
        if replace_existing:
            merged = tuple(models)
        else:
            by_name = {model.name: model for model in self.models}
            for model in models:
                by_name[model.name] = model
            merged = tuple(by_name.values())
        return replace(self, name=name or self.name, models=merged)

    def add_models(
        self,
        *models: ModelProfile,
        name: str | None = None,
    ) -> EvaluationProfile:
        return self.with_models(*models, name=name, replace_existing=False)

    def only_models(
        self,
        *models: ModelProfile,
        name: str | None = None,
    ) -> EvaluationProfile:
        return self.with_models(*models, name=name, replace_existing=True)


def model_profile(
    provider: str,
    model_id: str,
    *,
    name: str | None = None,
    api_key_env: str | None = None,
    dtype: str = "bfloat16",
    load_in_4bit: bool | None = None,
    generation: dict[str, Any] | None = None,
    notes: str = "",
) -> ModelProfile:
    """Create a model profile with one user-facing function."""

    provider_key = normalize_provider(provider)
    if provider_key == "huggingface":
        return huggingface_model_profile(
            hf_repo=model_id,
            name=name,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            generation=generation,
            notes=notes,
        )
    return closed_source_model_profile(
        provider=provider_key,
        model_id=model_id,
        name=name,
        api_key_env=api_key_env,
        generation=generation,
        notes=notes,
    )


def closed_source_model_profile(
    *,
    provider: str,
    model_id: str,
    name: str | None = None,
    api_key_env: str | None = None,
    generation: dict[str, Any] | None = None,
    notes: str = "",
) -> ModelProfile:
    """Create a common profile for a provider-API model adapter."""

    provider_key = normalize_provider(provider)
    spec = get_provider(provider_key)
    env = (api_key_env,) if api_key_env else spec.required_env
    return ModelProfile(
        name=name or model_id,
        provider=provider_key,
        model_id=model_id,
        adapter="provider_api",
        env=tuple(env),
        generation=dict(generation or {}),
        notes=notes,
    )


def huggingface_model_profile(
    *,
    hf_repo: str,
    name: str | None = None,
    dtype: str = "bfloat16",
    load_in_4bit: bool | None = None,
    generation: dict[str, Any] | None = None,
    notes: str = "",
) -> ModelProfile:
    """Create a common profile for a local/open-weight HuggingFace adapter."""

    options = {"dtype": dtype, **dict(generation or {})}
    if load_in_4bit is not None:
        options["load_in_4bit"] = load_in_4bit
    return ModelProfile(
        name=name or sanitize_dataset_name(hf_repo),
        provider="huggingface",
        model_id=hf_repo,
        adapter="huggingface_local",
        hf_repo=hf_repo,
        generation=options,
        notes=notes,
    )


def dataset_profile_from_hf(
    dataset_ref: str,
    *,
    name: str | None = None,
    split: str = "validation",
    facets: tuple[str, ...] | list[str] = ("option_order",),
    config: str | None = None,
    revision: str | None = None,
    license: str | None = None,
    filters: dict[str, Any] | None = None,
    notes: str = "Review generated profile before inference.",
) -> DatasetProfile:
    """Create a dataset profile from a HuggingFace dataset ID or URL."""

    hf_repo = normalize_hf_dataset_id(dataset_ref)
    return DatasetProfile(
        name=name or sanitize_dataset_name(hf_repo),
        hf_repo=hf_repo,
        split=split,
        facets=tuple(facets),
        config=config,
        revision=revision,
        license=license,
        filters=dict(filters or {}),
        notes=notes,
    )


def hf_dataset(
    dataset_ref: str,
    *,
    name: str | None = None,
    split: str = "validation",
    facets: tuple[str, ...] | list[str] = ("option_order",),
    config: str | None = None,
    revision: str | None = None,
    license: str | None = None,
    filters: dict[str, Any] | None = None,
    notes: str = "Review generated profile before inference.",
) -> DatasetProfile:
    """Create a HuggingFace dataset profile."""

    return dataset_profile_from_hf(
        dataset_ref,
        name=name,
        split=split,
        facets=facets,
        config=config,
        revision=revision,
        license=license,
        filters=filters,
        notes=notes,
    )


def paper_profile(
    *,
    config_dir: str | Path = "configs",
    model_config: str | Path | None = None,
    dataset_config: str | Path | None = None,
    models: NameSelection = None,
    datasets: NameSelection = None,
    k: int | None = None,
    seed: int | None = None,
    name: str = "facet-probe-paper-v0.0.1",
) -> EvaluationProfile:
    """Load the paper benchmark profile with shorter argument names."""

    return paper_evaluation_profile(
        config_dir=config_dir,
        model_config=model_config,
        dataset_config=dataset_config,
        model_names=_normalize_name_selection(models),
        dataset_names=_normalize_name_selection(datasets),
        k=k,
        seed=seed,
        name=name,
    )


def judge_profile(
    *,
    config_dir: str | Path = "configs",
    judge_config: str | Path | None = None,
    name: str = "mixed-semantic-primary",
) -> ModelProfile:
    """Load a named paper judge profile from the release model config."""

    config_path = _resolve_config_dir(config_dir)
    judges_path = Path(judge_config) if judge_config else config_path / "models.yaml"
    config = _load_yaml(judges_path)
    judge_specs = config.get("judges", {})
    if name not in judge_specs:
        raise KeyError(f"unknown judge profile {name!r}; known: {sorted(judge_specs)}")
    defaults = dict(config.get("generation_defaults", {}))
    return _model_profile_from_config(name, judge_specs[name], defaults)


def paper_evaluation_profile(
    *,
    config_dir: str | Path = "configs",
    model_config: str | Path | None = None,
    dataset_config: str | Path | None = None,
    model_names: tuple[str, ...] | list[str] | None = None,
    dataset_names: tuple[str, ...] | list[str] | None = None,
    k: int | None = None,
    seed: int | None = None,
    name: str = "facet-probe-paper-v0.0.1",
) -> EvaluationProfile:
    """Load the paper benchmark profile from the release config files."""

    config_path = _resolve_config_dir(config_dir)
    models_path = Path(model_config) if model_config else config_path / "models.yaml"
    datasets_path = Path(dataset_config) if dataset_config else config_path / "datasets.yaml"
    models_yaml = _load_yaml(models_path)
    datasets_yaml = _load_yaml(datasets_path)
    model_specs = models_yaml.get("models", {})
    dataset_specs = datasets_yaml.get("datasets", {})
    selected_models = _select_keys(model_specs, model_names, "model")
    selected_datasets = _select_keys(dataset_specs, dataset_names, "dataset")
    defaults = dict(models_yaml.get("generation_defaults", {}))
    if k is not None:
        defaults["k_orderings"] = k
    if seed is not None:
        defaults["seed"] = seed
    k_orderings = int(defaults.get("k_orderings", 6))
    profile_seed = int(
        seed if seed is not None else datasets_yaml.get("seed", defaults.get("seed", 42))
    )

    return EvaluationProfile(
        name=name,
        models=tuple(
            _model_profile_from_config(model_name, model_specs[model_name], defaults)
            for model_name in selected_models
        ),
        datasets=tuple(
            _dataset_profile_from_config(dataset_name, dataset_specs[dataset_name])
            for dataset_name in selected_datasets
        ),
        k_orderings=k_orderings,
        seed=profile_seed,
        generation_defaults=defaults,
        config_paths=(
            str(models_path),
            str(datasets_path),
            str(config_path / "facets.yaml"),
        ),
        notes=(
            "Uses the public v0.0.1 paper benchmark configs.",
            "Inference adapters must load upstream datasets and model/API outputs at runtime.",
        ),
    )


def normalize_provider(provider: str) -> str:
    key = provider.strip().lower().replace("_", "-")
    aliases = {
        "gemini": "google",
        "google": "google",
        "openai": "openai",
        "anthropic": "anthropic",
        "claude": "anthropic",
        "huggingface": "huggingface",
        "hf": "huggingface",
        "openai-compatible": "openai-compatible",
        "mock": "mock",
    }
    return aliases.get(key, key)


def _resolve_config_dir(config_dir: str | Path) -> ConfigPath:
    path = Path(config_dir)
    if path.exists() or str(config_dir) != "configs":
        return path
    return release_data_root() / "configs"


def _load_yaml(path: ConfigPath) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _normalize_name_selection(selected: NameSelection) -> tuple[str, ...] | None:
    if selected is None:
        return None
    if isinstance(selected, str):
        return (selected,)
    return tuple(selected)


def _select_keys(
    mapping: dict[str, Any],
    selected: tuple[str, ...] | list[str] | None,
    kind: str,
) -> tuple[str, ...]:
    if selected is None:
        return tuple(mapping)
    missing = [name for name in selected if name not in mapping]
    if missing:
        raise KeyError(f"unknown {kind} profile(s): {missing}; known: {sorted(mapping)}")
    return tuple(selected)


def _model_profile_from_config(
    name: str,
    spec: dict[str, Any],
    defaults: dict[str, Any],
) -> ModelProfile:
    generation = dict(defaults)
    generation.update(
        {key: value for key, value in spec.items() if key not in _MODEL_METADATA_KEYS}
    )
    if "hf_repo" in spec:
        return huggingface_model_profile(
            hf_repo=str(spec["hf_repo"]),
            name=name,
            dtype=str(generation.pop("dtype", "bfloat16")),
            load_in_4bit=generation.pop("load_in_4bit", None),
            generation=generation,
            notes=str(spec.get("note", "")),
        )
    provider = normalize_provider(str(spec.get("provider", "")))
    return closed_source_model_profile(
        provider=provider,
        model_id=str(spec.get("api_model_id", name)),
        name=name,
        api_key_env=str(spec["api_key_env"]) if spec.get("api_key_env") else None,
        generation=generation,
        notes=str(spec.get("note", "")),
    )


def _dataset_profile_from_config(name: str, spec: dict[str, Any]) -> DatasetProfile:
    audited_n = spec.get("audited_n", spec.get("audited_n_clean", spec.get("audited_n_raw")))
    split = spec.get("split")
    if split is None:
        split = "file-backed" if spec.get("files") else "unspecified"
    return DatasetProfile(
        name=name,
        hf_repo=str(spec["hf_repo"]),
        split=str(split),
        facets=tuple(spec.get("facets", ())),
        config=spec.get("config"),
        revision=spec.get("revision"),
        license=spec.get("license"),
        audited_n=audited_n,
        filters=dict(spec.get("filters", {})),
        notes=str(spec.get("notes", "")),
    )
