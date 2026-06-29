"""HuggingFace dataset inspection helpers.

The inspector is deliberately human-in-the-loop. It reads dataset metadata,
feature schemas, and optional row shapes, then suggests existing Facet-Probe
facets and a starter dataset spec. It does not store upstream examples.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

from facet_probe.datasets import infer_candidate_facets


@dataclass(frozen=True)
class FeatureSummary:
    path: str
    dtype: str


@dataclass(frozen=True)
class FieldProfile:
    path: str
    observed_types: tuple[str, ...]
    present_count: int
    list_min_len: int | None = None
    list_max_len: int | None = None


@dataclass(frozen=True)
class HFInspection:
    dataset_id: str
    config: str | None
    split: str | None
    revision: str | None
    license: str | None
    tags: tuple[str, ...]
    task_categories: tuple[str, ...]
    modalities: tuple[str, ...]
    configs: tuple[str, ...]
    splits: tuple[str, ...]
    features: tuple[FeatureSummary, ...]
    sample_row_count: int
    sample_profiles: tuple[FieldProfile, ...]
    candidate_facets: tuple[str, ...]
    warnings: tuple[str, ...]
    starter_dataset_spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_hf_dataset_id(value: str) -> str:
    """Normalize a HuggingFace dataset ID or URL to ``org/name`` form."""

    text = value.strip()
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        parts = [part for part in parsed.path.split("/") if part]
        if parts and parts[0] == "datasets":
            parts = parts[1:]
        if len(parts) >= 2:
            return "/".join(parts[:2])
    return text.removeprefix("datasets/").strip("/")


def sanitize_dataset_name(dataset_id: str) -> str:
    name = dataset_id.split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name or "new_dataset"


def flatten_feature_schema(schema: Any, *, prefix: str = "") -> tuple[FeatureSummary, ...]:
    """Flatten HF ``Features``-like objects into ``path, dtype`` summaries."""

    if schema is None:
        return ()

    if _is_mapping_like(schema):
        out: list[FeatureSummary] = []
        for key, value in schema.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.extend(flatten_feature_schema(value, prefix=child_prefix))
        return tuple(out)

    if isinstance(schema, (list, tuple)):
        if not schema:
            return (FeatureSummary(prefix or "[]", "empty_sequence"),)
        if len(schema) == 1:
            return flatten_feature_schema(schema[0], prefix=f"{prefix}[]")
        return tuple(
            item
            for idx, value in enumerate(schema)
            for item in flatten_feature_schema(value, prefix=f"{prefix}[{idx}]")
        )

    for attr in ("feature", "features"):
        child = getattr(schema, attr, None)
        if child is not None and child is not schema:
            child_prefix = f"{prefix}[]" if attr == "feature" else prefix
            child_items = flatten_feature_schema(child, prefix=child_prefix)
            if child_items:
                return child_items

    return (FeatureSummary(prefix or "<root>", _feature_dtype(schema)),)


def summarize_sample_rows(rows: list[dict[str, Any]]) -> tuple[FieldProfile, ...]:
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    present_counts: Counter[str] = Counter()
    list_lengths: dict[str, list[int]] = defaultdict(list)

    for row in rows:
        for path, value in _walk_row_shape(row):
            present_counts[path] += 1
            type_counts[path][type(value).__name__] += 1
            if isinstance(value, (list, tuple)):
                list_lengths[path].append(len(value))

    profiles: list[FieldProfile] = []
    for path in sorted(type_counts):
        lengths = list_lengths.get(path, [])
        profiles.append(
            FieldProfile(
                path=path,
                observed_types=tuple(sorted(type_counts[path])),
                present_count=present_counts[path],
                list_min_len=min(lengths) if lengths else None,
                list_max_len=max(lengths) if lengths else None,
            )
        )
    return tuple(profiles)


def build_hf_inspection(
    *,
    dataset_id: str,
    config: str | None = None,
    split: str | None = None,
    revision: str | None = None,
    license: str | None = None,
    tags: list[str] | tuple[str, ...] = (),
    task_categories: list[str] | tuple[str, ...] = (),
    modalities: list[str] | tuple[str, ...] = (),
    configs: list[str] | tuple[str, ...] = (),
    splits: list[str] | tuple[str, ...] = (),
    features: Any = None,
    sample_rows: list[dict[str, Any]] | None = None,
    warnings: list[str] | tuple[str, ...] = (),
) -> HFInspection:
    dataset_id = normalize_hf_dataset_id(dataset_id)
    feature_summaries = flatten_feature_schema(features)
    sample_profiles = summarize_sample_rows(sample_rows or [])
    feature_names = {item.path for item in feature_summaries}
    feature_names.update(profile.path for profile in sample_profiles)
    detected_modalities = _infer_modalities(
        feature_summaries=feature_summaries,
        sample_profiles=sample_profiles,
        modalities=modalities,
        task_categories=task_categories,
        tags=tags,
    )
    facet_tasks = tuple(dict.fromkeys([*task_categories, *tags]))
    candidate_facets = infer_candidate_facets(
        feature_names=feature_names,
        modalities=detected_modalities,
        task_categories=facet_tasks,
    )
    all_warnings = list(warnings)
    all_warnings.extend(
        _inspection_warnings(
            license=license,
            candidate_facets=candidate_facets,
            feature_names=feature_names,
            modalities=detected_modalities,
            sample_profiles=sample_profiles,
        )
    )
    spec = starter_dataset_spec(
        dataset_id=dataset_id,
        config=config,
        split=split,
        license=license,
        candidate_facets=candidate_facets,
        warnings=all_warnings,
    )
    return HFInspection(
        dataset_id=dataset_id,
        config=config,
        split=split,
        revision=revision,
        license=license,
        tags=tuple(dict.fromkeys(tags)),
        task_categories=tuple(dict.fromkeys(task_categories)),
        modalities=detected_modalities,
        configs=tuple(configs),
        splits=tuple(splits),
        features=feature_summaries,
        sample_row_count=len(sample_rows or []),
        sample_profiles=sample_profiles,
        candidate_facets=candidate_facets,
        warnings=tuple(dict.fromkeys(all_warnings)),
        starter_dataset_spec=spec,
    )


def starter_dataset_spec(
    *,
    dataset_id: str,
    config: str | None,
    split: str | None,
    license: str | None,
    candidate_facets: tuple[str, ...],
    warnings: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    key = sanitize_dataset_name(dataset_id)
    spec: dict[str, Any] = {
        "hf_repo": dataset_id,
        "split": split or "unspecified",
        "license": license or "REVIEW_LICENSE",
        "audited_n": "REVIEW_N",
        "facets": list(candidate_facets) or ["REVIEW_FACET"],
        "notes": "Starter spec from facet-probe inspect-hf; review before inference.",
    }
    if config:
        spec["config"] = config
    if warnings:
        spec["review_warnings"] = list(dict.fromkeys(warnings))
    return {key: spec}


def inspect_hf_dataset(
    dataset_ref: str,
    *,
    config: str | None = None,
    split: str | None = None,
    revision: str | None = None,
    sample: int = 20,
    streaming: bool = True,
) -> HFInspection:
    """Inspect a HuggingFace dataset.

    Requires the optional ``hf`` extra. Network access and HuggingFace
    authentication are governed by the caller's environment.
    """

    dataset_id = normalize_hf_dataset_id(dataset_ref)
    try:
        from datasets import (  # type: ignore
            get_dataset_config_names,
            get_dataset_split_names,
            load_dataset,
            load_dataset_builder,
        )
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by user env
        raise RuntimeError(
            "Install HuggingFace support with `pip install -e '.[hf]'` "
            "or `bash setup.sh uv --extras hf`."
        ) from exc

    warnings: list[str] = []
    tags: list[str] = []
    task_categories: list[str] = []
    modalities: list[str] = []
    license_name: str | None = None

    try:
        info = HfApi().dataset_info(dataset_id, revision=revision)
        tags = list(getattr(info, "tags", None) or [])
        card_data = getattr(info, "cardData", None) or {}
        license_name = _metadata_value(card_data, "license")
        task_categories = _metadata_list(card_data, "task_categories")
        modalities = _metadata_list(card_data, "modalities")
    except Exception as exc:  # pragma: no cover - depends on remote service
        warnings.append(f"dataset card metadata unavailable: {exc}")

    try:
        configs = tuple(get_dataset_config_names(dataset_id, revision=revision))
    except Exception as exc:  # pragma: no cover - depends on remote service
        configs = ()
        warnings.append(f"config list unavailable: {exc}")

    selected_config = config or (configs[0] if len(configs) == 1 else None)
    try:
        split_kwargs = {"revision": revision} if revision else {}
        splits = tuple(get_dataset_split_names(dataset_id, selected_config, **split_kwargs))
    except Exception as exc:  # pragma: no cover - depends on remote service
        splits = ()
        warnings.append(f"split list unavailable: {exc}")

    selected_split = split or _default_split(splits)

    features = None
    try:
        args = [dataset_id]
        if selected_config:
            args.append(selected_config)
        builder_kwargs = {"revision": revision} if revision else {}
        builder = load_dataset_builder(*args, **builder_kwargs)
        features = builder.info.features
        if not splits and getattr(builder.info, "splits", None):
            splits = tuple(builder.info.splits.keys())
            selected_split = split or _default_split(splits)
        if license_name is None:
            license_name = getattr(builder.info, "license", None) or None
    except Exception as exc:  # pragma: no cover - depends on remote service
        warnings.append(f"feature schema unavailable: {exc}")

    sample_rows: list[dict[str, Any]] = []
    if sample > 0 and selected_split:
        try:
            args = [dataset_id]
            if selected_config:
                args.append(selected_config)
            kwargs: dict[str, Any] = {
                "split": selected_split,
                "streaming": streaming,
            }
            if revision:
                kwargs["revision"] = revision
            ds = load_dataset(*args, **kwargs)
            for idx, row in enumerate(ds):
                if idx >= sample:
                    break
                sample_rows.append(row)
        except Exception as exc:  # pragma: no cover - depends on remote service
            warnings.append(f"sample-row inspection unavailable: {exc}")

    return build_hf_inspection(
        dataset_id=dataset_id,
        config=selected_config,
        split=selected_split,
        revision=revision,
        license=license_name,
        tags=tags,
        task_categories=task_categories,
        modalities=modalities,
        configs=configs,
        splits=splits,
        features=features,
        sample_rows=sample_rows,
        warnings=warnings,
    )


def _infer_modalities(
    *,
    feature_summaries: tuple[FeatureSummary, ...],
    sample_profiles: tuple[FieldProfile, ...],
    modalities: list[str] | tuple[str, ...],
    task_categories: list[str] | tuple[str, ...],
    tags: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    out = [str(item).lower() for item in modalities]
    signals = " ".join(
        [
            *(f"{item.path} {item.dtype}" for item in feature_summaries),
            *(f"{item.path} {' '.join(item.observed_types)}" for item in sample_profiles),
            *map(str, task_categories),
            *map(str, tags),
        ]
    ).lower()
    if any(token in signals for token in ["image", "visual", "vision"]):
        out.append("image")
    if any(token in signals for token in ["audio", "speech"]):
        out.append("audio")
    if "table" in signals:
        out.append("table")
    if any(token in signals for token in ["text", "question", "context", "answer"]):
        out.append("text")
    return tuple(dict.fromkeys(item for item in out if item))


def _inspection_warnings(
    *,
    license: str | None,
    candidate_facets: tuple[str, ...],
    feature_names: set[str],
    modalities: tuple[str, ...],
    sample_profiles: tuple[FieldProfile, ...],
) -> list[str]:
    warnings: list[str] = []
    lowered = {name.lower() for name in feature_names}
    if not candidate_facets:
        warnings.append("no existing Facet-Probe facet could be inferred automatically")
    if "option_order" in candidate_facets and not (
        {"answer", "answers", "answerkey", "answer_index", "label", "gold"} & lowered
    ):
        warnings.append("review gold-label field before using option_order")
    if "document_rank_order" in candidate_facets and "rank" not in lowered:
        warnings.append("review retrieved-document rank semantics")
    if "image" in modalities:
        warnings.append("review image loading/rendering and redistribution terms")
    if license and any(marker in license.lower() for marker in ["nc", "non-commercial"]):
        warnings.append("license appears to restrict commercial use")
    if not sample_profiles:
        warnings.append("no sample rows inspected; run with --sample for row-shape checks")
    return warnings


def _walk_row_shape(value: Any, *, prefix: str = "", depth: int = 0):
    path = prefix or "<root>"
    yield path, value
    if depth >= 3:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_row_shape(child, prefix=child_prefix, depth=depth + 1)
    elif isinstance(value, (list, tuple)) and value:
        yield from _walk_row_shape(value[0], prefix=f"{prefix}[]", depth=depth + 1)


def _is_mapping_like(value: Any) -> bool:
    return not isinstance(value, (str, bytes)) and hasattr(value, "items")


def _feature_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        return str(dtype)
    return type(value).__name__


def _metadata_value(card_data: Any, key: str) -> str | None:
    if isinstance(card_data, dict):
        value = card_data.get(key)
    else:
        value = getattr(card_data, key, None)
    if isinstance(value, list):
        return str(value[0]) if value else None
    return None if value is None else str(value)


def _metadata_list(card_data: Any, key: str) -> list[str]:
    if isinstance(card_data, dict):
        value = card_data.get(key) or []
    else:
        value = getattr(card_data, key, None) or []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _default_split(splits: tuple[str, ...]) -> str | None:
    for candidate in ("test", "validation", "dev", "train"):
        if candidate in splits:
            return candidate
    return splits[0] if splits else None
