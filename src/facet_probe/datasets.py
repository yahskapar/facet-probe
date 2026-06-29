"""Dataset registry and HuggingFace-oriented extension helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_repo: str
    split: str
    primary_facets: tuple[str, ...]
    audited_n: int | str
    license: str
    config: str | None = None
    notes: str = ""
    filters: dict[str, Any] = field(default_factory=dict)


PAPER_DATASETS: dict[str, DatasetSpec] = {
    "mmlu_pro": DatasetSpec(
        name="mmlu_pro",
        hf_repo="TIGER-Lab/MMLU-Pro",
        split="test",
        primary_facets=("option_order",),
        audited_n=200,
        license="MIT",
        filters={"min_choices": 4},
    ),
    "commonsenseqa": DatasetSpec(
        name="commonsenseqa",
        hf_repo="tau/commonsense_qa",
        split="validation",
        primary_facets=("option_order",),
        audited_n=200,
        license="MIT",
    ),
    "mathvision": DatasetSpec(
        name="mathvision",
        hf_repo="MathLLMs/MathVision",
        split="testmini",
        primary_facets=("option_order",),
        audited_n=190,
        license="CC-BY-4.0",
        filters={"min_choices": 3, "requires_image": True},
    ),
    "hotpotqa": DatasetSpec(
        name="hotpotqa",
        hf_repo="hotpotqa/hotpot_qa",
        split="validation",
        config="default",
        primary_facets=("evidence_chunk_order", "document_rank_order"),
        audited_n=199,
        license="CC-BY-SA-4.0",
        notes="Distractor-augmented validation split; retrieved evidence replayed verbatim.",
    ),
    "musique": DatasetSpec(
        name="musique",
        hf_repo="dgslibisey/MuSiQue",
        split="validation",
        primary_facets=("evidence_chunk_order",),
        audited_n=200,
        license="CC-BY-4.0",
    ),
    "medxpertqa": DatasetSpec(
        name="medxpertqa",
        hf_repo="TsinghuaC3I/MedXpertQA",
        split="test",
        config="MM",
        primary_facets=("evidence_chunk_order",),
        audited_n=150,
        license="CC-BY-NC-4.0",
        notes="Retrieved-evidence variant; used for non-commercial academic evaluation.",
    ),
    "multihop_rag": DatasetSpec(
        name="multihop_rag",
        hf_repo="yixuantt/MultiHopRAG",
        split="train",
        config="MultiHopRAG",
        primary_facets=("document_rank_order",),
        audited_n=171,
        license="MIT",
    ),
    "mantis_eval": DatasetSpec(
        name="mantis_eval",
        hf_repo="TIGER-Lab/Mantis-Eval",
        split="test",
        primary_facets=("image_set_order",),
        audited_n="70 raw / 18 clean",
        license="Apache-2.0",
        filters={"min_images": 3, "question_type": "multi-choice"},
        notes="Clean summaries apply the position-reference screen.",
    ),
    "medframeqa": DatasetSpec(
        name="medframeqa",
        hf_repo="SuhaoYu1020/MedFrameQA",
        split="test",
        primary_facets=("image_set_order",),
        audited_n="200 raw / 195 clean",
        license="CC-BY-NC-4.0",
        notes="Used for non-commercial academic evaluation; clean summaries apply the screen.",
    ),
    "mramg": DatasetSpec(
        name="mramg",
        hf_repo="MRAMG/MRAMG-Bench",
        split="recipe_mqa.jsonl",
        primary_facets=("mixed_modality_order",),
        audited_n=197,
        license="CC-BY-4.0",
        notes="MRAMG-Recipe after image-load filter.",
    ),
    "mmdocrag": DatasetSpec(
        name="mmdocrag",
        hf_repo="MMDocIR/MMDocRAG",
        split="dev_15.jsonl",
        primary_facets=("mixed_modality_order",),
        audited_n=200,
        license="Apache-2.0",
    ),
    "mmqa": DatasetSpec(
        name="mmqa",
        hf_repo="TableQAKit/MMQA",
        split="dev",
        primary_facets=("mixed_modality_order",),
        audited_n=200,
        license="MIT",
        notes="MultiModalQA short-factoid subset provides the mixed-modality gold anchor.",
    ),
}


def list_paper_datasets() -> list[DatasetSpec]:
    return [PAPER_DATASETS[name] for name in sorted(PAPER_DATASETS)]


def get_dataset(name: str) -> DatasetSpec:
    try:
        return PAPER_DATASETS[name]
    except KeyError as exc:
        known = sorted(PAPER_DATASETS)
        raise KeyError(f"unknown dataset {name!r}; known datasets: {known}") from exc


def infer_candidate_facets(
    *,
    feature_names: set[str] | list[str],
    modalities: set[str] | list[str] = (),
    task_categories: set[str] | list[str] = (),
) -> tuple[str, ...]:
    """Suggest ordering facets from dataset-card or feature metadata.

    This is intentionally conservative. Dataset-specific specs should override
    these defaults when the task has unusual structure.
    """

    features = _feature_aliases(feature_names)
    mods = {m.lower() for m in modalities}
    tasks = {t.lower() for t in task_categories}
    out: list[str] = []

    if {"choices", "choice", "options", "option", "answerkey", "answer_index"} & features:
        out.append("option_order")
    if (
        {"context", "paragraphs", "paragraph", "evidence", "evidence_list", "passages", "passage"}
        & features
    ):
        out.append("evidence_chunk_order")
    if {"documents", "document", "retrieved_documents", "evidence_list", "rank"} & features:
        out.append("document_rank_order")
    if {"images", "image", "image_list", "frames", "frame"} & features or "image" in mods:
        out.append("image_set_order")
    if (
        ("image" in mods and ("text" in mods or "table" in mods))
        or "multimodal" in tasks
        or "visual-question-answering" in tasks
    ):
        out.append("mixed_modality_order")
    return tuple(dict.fromkeys(out))


def _feature_aliases(feature_names: set[str] | list[str]) -> set[str]:
    aliases: set[str] = set()
    for raw in feature_names:
        text = str(raw).strip().lower()
        if not text:
            continue
        aliases.add(text)
        normalized = (
            text.replace("[]", "")
            .replace("[", ".")
            .replace("]", "")
            .replace("/", ".")
            .replace("-", "_")
        )
        aliases.add(normalized)
        for part in normalized.split("."):
            if part:
                aliases.add(part)
        for suffix in ("_list", "_text", "_ids", "_id"):
            if normalized.endswith(suffix):
                aliases.add(normalized.removesuffix(suffix))
    return aliases


def load_hf_dataset_rows(name: str, *, limit: int | None = None, streaming: bool = True):
    """Yield raw rows from a registered HuggingFace dataset.

    This helper keeps dynamic dataset loading out of the core release path. It
    imports datasets lazily so the package can be installed without HF extras.
    """

    from datasets import load_dataset  # type: ignore

    spec = get_dataset(name)
    args = [spec.hf_repo]
    if spec.config:
        args.append(spec.config)
    ds = load_dataset(*args, split=spec.split, streaming=streaming)
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        yield row
