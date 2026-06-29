"""Paper-specific dataset loaders for reproducible Facet-Probe runs.

These loaders port the public, dataset-normalization portions of the older
MMIOS evaluation scripts into the Facet-Probe package. They intentionally keep
upstream text/images in the runtime environment and emit normalized
``RuntimeExample`` objects for the runner.
"""

from __future__ import annotations

import gzip
import inspect
import io
import json
import os
import random
import re
import time
import warnings
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from facet_probe.facets import get_facet
from facet_probe.profiles import DatasetProfile
from facet_probe.schema import AuditItem, Component
from facet_probe.scoring import normalize_answer
from facet_probe.templates import content_ref

SEED = 42
MMLU_PRO_TEST_PARQUET = (
    "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/"
    "data/test-00000-of-00001.parquet"
)


def load_paper_dataset_examples(
    dataset: DatasetProfile,
    *,
    target: int | None,
    streaming: bool = True,
) -> list[Any] | None:
    """Return paper-normalized RuntimeExamples for a registered dataset.

    ``None`` means no paper-specific loader is registered, so the caller may
    fall back to generic HuggingFace field inference.
    """

    out = []
    known = False
    for facet in dataset.facets:
        loader = _LOADERS.get((facet, dataset.name))
        if loader is None:
            continue
        known = True
        out.extend(_call_loader(loader, dataset, target, streaming=streaming))
    return out if known else None


def _runtime_example_cls():
    from facet_probe.runner import RuntimeExample

    return RuntimeExample


def _make_mcq_example(
    *,
    dataset: DatasetProfile,
    facet: str,
    raw_id: str,
    question: str,
    choices: Sequence[str],
    gold_idx: int,
    fixed_images: Sequence[Any] = (),
    metadata: Mapping[str, Any] | None = None,
):
    runtime_example = _runtime_example_cls()
    components = tuple(
        Component(
            component_id=f"choice_{idx}",
            kind="choice",
            content_ref=content_ref(dataset.name, raw_id, "choices", idx),
            label=chr(ord("A") + idx),
        )
        for idx in range(len(choices))
    )
    question_ref = content_ref(dataset.name, raw_id, "question")
    fixed_components = tuple(
        Component(
            component_id=f"fixed_image_{idx}",
            kind="image",
            content_ref=content_ref(dataset.name, raw_id, "image", idx),
            label=f"Image {idx + 1}",
            metadata={"fixed": True},
        )
        for idx in range(len(fixed_images))
    )
    item = AuditItem(
        item_id=f"{dataset.name}::{raw_id}",
        dataset=dataset.name,
        components=components,
        question_ref=question_ref,
        choices=tuple(str(choice) for choice in choices),
        gold=str(gold_idx),
        metadata=dict(metadata or {}),
    )
    content = {question_ref: question}
    content.update(
        {
            component.content_ref: item.choices[idx]
            for idx, component in enumerate(components)
        }
    )
    content.update(
        {
            component.content_ref: fixed_images[idx]
            for idx, component in enumerate(fixed_components)
        }
    )
    return runtime_example(
        facet=facet,
        item=item,
        question=question,
        content=content,
        score_kind="option_content_idx",
        gold_normalized=str(gold_idx),
        fixed_components=fixed_components,
    )


def _make_sequence_example(
    *,
    dataset: DatasetProfile,
    facet: str,
    raw_id: str,
    question: str,
    components_text: Sequence[tuple[str, str]],
    gold: str,
    kind: str = "text",
    labels: Sequence[str] | None = None,
    score_kind: str | None = None,
):
    runtime_example = _runtime_example_cls()
    score = score_kind or get_facet(facet).score_kind
    components = tuple(
        Component(
            component_id=f"{kind}_{idx}",
            kind="document" if kind == "document" else "text",
            content_ref=content_ref(dataset.name, raw_id, kind, idx),
            label=(labels[idx] if labels else title),
        )
        for idx, (title, _body) in enumerate(components_text)
    )
    question_ref = content_ref(dataset.name, raw_id, "question")
    item = AuditItem(
        item_id=f"{dataset.name}::{raw_id}",
        dataset=dataset.name,
        components=components,
        question_ref=question_ref,
        gold=gold,
    )
    content = {question_ref: question}
    for idx, component in enumerate(components):
        title, body = components_text[idx]
        content[component.content_ref] = f"{title}: {body}" if title else body
    return runtime_example(
        facet=facet,
        item=item,
        question=question,
        content=content,
        score_kind=score,
        gold_normalized=normalize_answer(score, gold),
    )


def _make_image_set_example(
    *,
    dataset: DatasetProfile,
    raw_id: str,
    question: str,
    images: Sequence[Any],
    choices: Sequence[str],
    gold_letter: str,
    metadata: Mapping[str, Any] | None = None,
):
    runtime_example = _runtime_example_cls()
    components = tuple(
        Component(
            component_id=f"image_{idx}",
            kind="image",
            content_ref=content_ref(dataset.name, raw_id, "images", idx),
            label=f"Image {idx + 1}",
        )
        for idx in range(len(images))
    )
    question_ref = content_ref(dataset.name, raw_id, "question")
    item = AuditItem(
        item_id=f"{dataset.name}::{raw_id}",
        dataset=dataset.name,
        components=components,
        question_ref=question_ref,
        choices=tuple(str(choice) for choice in choices),
        gold=gold_letter,
        metadata=dict(metadata or {}),
    )
    content = {question_ref: question}
    content.update({component.content_ref: images[idx] for idx, component in enumerate(components)})
    return runtime_example(
        facet="image_set_order",
        item=item,
        question=question,
        content=content,
        score_kind="mcq_letter",
        gold_normalized=gold_letter,
    )


def _norm_text(value: Any) -> str:
    return normalize_answer("exact_match", None if value is None else str(value)) or ""


def _shuffled_buffer(
    rows: Iterable[dict[str, Any]],
    *,
    target: int,
    multiplier: int,
    seed: int,
) -> list[dict[str, Any]]:
    buf = []
    for row in rows:
        buf.append(row)
        if len(buf) >= target * multiplier:
            break
    random.Random(seed).shuffle(buf)
    return buf


def _decode_image_field(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return value if hasattr(value, "save") and hasattr(value, "convert") else None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, Mapping):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        path = value.get("path")
        if path and Path(path).exists():
            return Image.open(path).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    return value if hasattr(value, "save") and hasattr(value, "convert") else None


def _load_dataset_once(*args, **kwargs):
    from datasets import load_dataset  # type: ignore

    return load_dataset(*args, **kwargs)


def _load_dataset(*args, **kwargs):
    attempts = max(1, int(os.environ.get("FACET_PROBE_HF_RETRIES", "3")))
    sleep_seconds = max(0.0, float(os.environ.get("FACET_PROBE_HF_RETRY_SLEEP", "2")))
    for attempt in range(1, attempts + 1):
        try:
            return _load_dataset_once(*args, **kwargs)
        except Exception as exc:
            if attempt >= attempts or not _is_transient_hf_error(exc):
                raise
            warnings.warn(
                (
                    "Facet-Probe: transient HuggingFace dataset load failure "
                    f"on attempt {attempt}/{attempts}: {exc}. Retrying..."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            if sleep_seconds:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError("unreachable HuggingFace retry state")


def _is_transient_hf_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "504",
        "gateway time-out",
        "gateway timeout",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "502",
        "503",
        "connection error",
        "connection reset",
    )
    return any(marker in text for marker in transient_markers)


def _load_mmlu_pro_test_dataset(*, streaming: bool):
    try:
        return _load_dataset("TIGER-Lab/MMLU-Pro", split="test", streaming=streaming)
    except Exception as exc:
        if not _is_transient_hf_error(exc):
            raise
        warnings.warn(
            (
                "Facet-Probe: HuggingFace timed out while listing TIGER-Lab/MMLU-Pro; "
                "falling back to the direct public test parquet file."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return _load_dataset(
            "parquet",
            data_files={"test": MMLU_PRO_TEST_PARQUET},
            split="test",
            streaming=streaming,
        )


Loader = Callable[..., list[Any]]


def _call_loader(
    loader: Loader,
    dataset: DatasetProfile,
    target: int | None,
    *,
    streaming: bool,
) -> list[Any]:
    try:
        parameters = inspect.signature(loader).parameters
    except (TypeError, ValueError):
        return loader(dataset, target, streaming=streaming)
    accepts_streaming = "streaming" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_streaming:
        return loader(dataset, target, streaming=streaming)
    return loader(dataset, target)

MIXED_MODALITY_SYSTEM = (
    "You are a careful multimodal reader. You will be given a set of evidence "
    "items, including text passages and images, followed by an open-ended "
    "question. Read every piece of evidence before answering. Respond with a "
    "concise free-form natural-language answer grounded in the evidence. Do "
    "not respond with a single letter or a multiple-choice label."
)
MIXED_MODALITY_ANSWER_INSTRUCTION = (
    "Write a concise free-form answer grounded in the evidence. Begin directly "
    "with the answer; do not include a preamble or a multiple-choice letter."
)


def load_option_order_mmlu_pro(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    n = target or 200
    ds = _load_mmlu_pro_test_dataset(streaming=streaming)
    rows = _shuffled_buffer(ds, target=n, multiplier=8, seed=SEED)
    out = []
    for row in rows:
        options = row.get("options")
        if not isinstance(options, list) or len(options) < 4:
            continue
        if row.get("answer_index") is None:
            continue
        out.append(
            _make_mcq_example(
                dataset=dataset,
                facet="option_order",
                raw_id=str(row.get("question_id", len(out))),
                question=str(row.get("question", "")),
                choices=options,
                gold_idx=int(row["answer_index"]),
                metadata={"category": row.get("category", ""), "n_choices": len(options)},
            )
        )
        if len(out) >= n:
            break
    return out


def load_option_order_commonsenseqa(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    n = target or 200
    ds = _load_dataset("tau/commonsense_qa", split="validation", streaming=streaming)
    rows = _shuffled_buffer(ds, target=n, multiplier=4, seed=SEED)
    out = []
    for row in rows:
        choices = row.get("choices") or {}
        labels = list(choices.get("label") or [])
        texts = list(choices.get("text") or [])
        answer_key = row.get("answerKey")
        if not labels or not texts or answer_key not in labels:
            continue
        out.append(
            _make_mcq_example(
                dataset=dataset,
                facet="option_order",
                raw_id=str(row.get("id", len(out))),
                question=str(row.get("question", "")),
                choices=texts,
                gold_idx=labels.index(answer_key),
            )
        )
        if len(out) >= n:
            break
    return out


def load_option_order_mathvision(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    n = target or 190
    ds = _load_dataset("MathLLMs/MathVision", split="testmini", streaming=streaming)
    out = []
    for row in ds:
        options = row.get("options")
        answer = str(row.get("answer", "")).strip().upper()
        if not isinstance(options, (list, tuple)) or len(options) < 3:
            continue
        if not (len(answer) == 1 and "A" <= answer <= "Z"):
            continue
        gold_idx = ord(answer) - ord("A")
        if gold_idx >= len(options):
            continue
        image = row.get("decoded_image") or row.get("image")
        fixed = [_decode_image_field(image)] if image is not None else []
        fixed = [img for img in fixed if img is not None]
        out.append(
            _make_mcq_example(
                dataset=dataset,
                facet="option_order",
                raw_id=str(row.get("id", row.get("question_id", len(out)))),
                question=str(row.get("question", "")),
                choices=list(options),
                gold_idx=gold_idx,
                fixed_images=fixed,
            )
        )
        if len(out) >= n:
            break
    return out


def _load_hotpotqa_val_ds(streaming: bool = True):
    kwargs = {
        "path": "hotpotqa/hotpot_qa",
        "name": "default",
        "split": "validation",
        "revision": "refs/convert/parquet",
    }
    try:
        return _load_dataset(**kwargs, streaming=streaming)
    except Exception:
        return _load_dataset(**kwargs, streaming=False)


def load_evidence_chunk_order_hotpotqa(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    n = target or 199
    rows = _shuffled_buffer(
        _load_hotpotqa_val_ds(streaming=streaming),
        target=n,
        multiplier=2,
        seed=SEED,
    )
    out = []
    for row in rows:
        context = row.get("context") or {}
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        chunks = []
        for title, sents in list(zip(titles, sentences, strict=False))[:6]:
            body = " ".join(str(sent) for sent in sents).strip()
            if body:
                chunks.append((str(title), body))
        if len(chunks) < 3:
            continue
        out.append(
            _make_sequence_example(
                dataset=dataset,
                facet="evidence_chunk_order",
                raw_id=str(row.get("id", len(out))),
                question=str(row.get("question", "")),
                components_text=chunks,
                gold=_norm_text(row.get("answer")),
                kind="text",
            )
        )
        if len(out) >= n:
            break
    return out


def load_evidence_chunk_order_musique(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    n = target or 200
    ds = _load_dataset("dgslibisey/MuSiQue", split="validation", streaming=streaming)
    rows = _shuffled_buffer(ds, target=n, multiplier=2, seed=SEED + 1)
    out = []
    for row in rows:
        paragraphs = row.get("paragraphs") or []
        supporting = [p for p in paragraphs if isinstance(p, Mapping) and p.get("is_supporting")]
        distractors = [
            p
            for p in paragraphs
            if isinstance(p, Mapping) and not p.get("is_supporting")
        ]
        kept = (supporting + distractors)[:6]
        chunks = [
            (str(p.get("title", "")), str(p.get("paragraph_text", "")))
            for p in kept
            if str(p.get("paragraph_text", "")).strip()
        ]
        if len(chunks) < 3:
            continue
        out.append(
            _make_sequence_example(
                dataset=dataset,
                facet="evidence_chunk_order",
                raw_id=str(row.get("id", len(out))),
                question=str(row.get("question", "")),
                components_text=chunks,
                gold=_norm_text(row.get("answer")),
                kind="text",
            )
        )
        if len(out) >= n:
            break
    return out


def load_document_rank_order_multihop_rag(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    n = target or 171
    ds = _load_dataset(
        "yixuantt/MultiHopRAG",
        "MultiHopRAG",
        split="train",
        streaming=streaming,
    )
    rows = _shuffled_buffer(ds, target=n, multiplier=3, seed=SEED + 2)
    out = []
    for row_idx, row in enumerate(rows):
        evidence = row.get("evidence_list") or []
        if isinstance(evidence, str):
            try:
                evidence = json.loads(evidence)
            except json.JSONDecodeError:
                evidence = []
        docs = []
        for ev in list(evidence)[:6]:
            if isinstance(ev, Mapping):
                title = ev.get("title") or ev.get("source") or ""
                body = ev.get("fact") or ev.get("body") or ev.get("snippet") or ""
                if body:
                    docs.append((str(title), str(body)))
            elif isinstance(ev, str) and ev.strip():
                docs.append(("", ev.strip()))
        if len(docs) < 2:
            continue
        out.append(
            _make_sequence_example(
                dataset=dataset,
                facet="document_rank_order",
                raw_id=f"{row_idx:04d}",
                question=str(row.get("query", "")),
                components_text=docs,
                gold=_norm_text(row.get("answer")),
                kind="document",
                labels=[f"Retrieved doc rank {idx + 1}" for idx in range(len(docs))],
            )
        )
        if len(out) >= n:
            break
    return out


def load_document_rank_order_hotpotqa(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    n = target or 199
    rows = _shuffled_buffer(
        _load_hotpotqa_val_ds(streaming=streaming),
        target=n,
        multiplier=4,
        seed=SEED + 7,
    )
    out = []
    for row in rows:
        context = row.get("context") or {}
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        support = set((row.get("supporting_facts") or {}).get("title") or [])
        ranked = []
        for title, sents in zip(titles, sentences, strict=False):
            if title in support:
                body = " ".join(str(sent) for sent in sents).strip()
                if body:
                    ranked.append((str(title), body))
        for title, sents in zip(titles, sentences, strict=False):
            if title not in support:
                body = " ".join(str(sent) for sent in sents).strip()
                if body and len(ranked) < 6:
                    ranked.append((str(title), body))
        if len(ranked) < 3:
            continue
        out.append(
            _make_sequence_example(
                dataset=dataset,
                facet="document_rank_order",
                raw_id=str(row.get("id", len(out))),
                question=str(row.get("question", "")),
                components_text=ranked,
                gold=_norm_text(row.get("answer")),
                kind="document",
                labels=[f"Retrieved doc rank {idx + 1}" for idx in range(len(ranked))],
            )
        )
        if len(out) >= n:
            break
    return out


def _make_evidence_mcq_example(
    *,
    dataset: DatasetProfile,
    raw_id: str,
    question: str,
    components_text: Sequence[tuple[str, str]],
    choices: Sequence[str],
    gold_idx: int,
    fixed_images: Sequence[Any] = (),
    metadata: Mapping[str, Any] | None = None,
):
    runtime_example = _runtime_example_cls()
    components = tuple(
        Component(
            component_id=f"evidence_{idx}",
            kind="text",
            content_ref=content_ref(dataset.name, raw_id, "evidence", idx),
            label=title,
        )
        for idx, (title, _body) in enumerate(components_text)
    )
    fixed_components = tuple(
        Component(
            component_id=f"fixed_image_{idx}",
            kind="image",
            content_ref=content_ref(dataset.name, raw_id, "image", idx),
            label=f"Image {idx + 1}",
            metadata={"fixed": True},
        )
        for idx in range(len(fixed_images))
    )
    question_ref = content_ref(dataset.name, raw_id, "question")
    item = AuditItem(
        item_id=f"{dataset.name}::{raw_id}",
        dataset=dataset.name,
        components=components,
        question_ref=question_ref,
        choices=tuple(str(choice) for choice in choices),
        gold=chr(ord("A") + gold_idx),
        metadata=dict(metadata or {}),
    )
    content = {question_ref: question}
    for idx, component in enumerate(components):
        title, body = components_text[idx]
        content[component.content_ref] = f"{title}: {body}" if title else body
    content.update(
        {
            component.content_ref: fixed_images[idx]
            for idx, component in enumerate(fixed_components)
        }
    )
    return runtime_example(
        facet="evidence_chunk_order",
        item=item,
        question=question,
        content=content,
        score_kind="mcq_letter",
        gold_normalized=chr(ord("A") + gold_idx),
        fixed_components=fixed_components,
    )


def load_evidence_chunk_order_medxpertqa(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    del streaming
    n = target or 150
    rows = _load_medxpertqa_records(dataset.hf_repo, dataset.config, dataset.split)
    usable = []
    for row in rows:
        images = [_decode_image_field(img) for img in _medxpert_images(row)]
        images = [img for img in images if img is not None]
        if not images:
            continue
        text = _medxpert_question_text(row)
        chunks = _split_medxpert_headers(text)
        if len(chunks) < 3:
            continue
        usable.append((row, chunks, images))
    random.Random(SEED).shuffle(usable)

    out = []
    for row, chunks, images in usable:
        three = _pick_three_medxpert_chunks(chunks)
        if three is None:
            continue
        try:
            choices, gold_idx = _medxpert_choices_and_answer(row)
        except ValueError:
            continue
        raw_id = str(row.get("id", row.get("question_id", len(out))))
        out.append(
            _make_evidence_mcq_example(
                dataset=dataset,
                raw_id=raw_id,
                question="Based on the evidence above, select the best answer.",
                components_text=[
                    ("Evidence A", three[0]),
                    ("Evidence B", three[1]),
                    ("Evidence C", three[2]),
                ],
                choices=choices,
                gold_idx=gold_idx,
                fixed_images=images[:1],
                metadata={
                    "specialty": row.get("specialty") or row.get("medical_task") or "",
                    "n_images": len(images),
                },
            )
        )
        if len(out) >= n:
            break
    return out


def _load_medxpertqa_records(repo: str, subset: str | None, split: str) -> list[dict[str, Any]]:
    try:
        if subset:
            loaded = _load_dataset(repo, subset, split=split, streaming=False)
        else:
            loaded = _load_dataset(repo, split=split, streaming=False)
        return [dict(row) for row in loaded]
    except TypeError as exc:
        if "dataclass" not in str(exc):
            raise
    return _load_medxpertqa_jsonl_records(repo, subset, split)


def _load_medxpertqa_jsonl_records(
    repo: str,
    subset: str | None,
    split: str,
) -> list[dict[str, Any]]:
    if not subset:
        raise RuntimeError("MedXpertQA fallback requires a subset/config such as 'MM'.")
    jsonl_path = _hf_download(repo, f"{subset}/{split}.jsonl")
    zip_path = _hf_download(repo, "images.zip")
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        image_index = {
            Path(name).name: name
            for name in zf.namelist()
            if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        }
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                records.append(_resolve_medxpert_image_refs(record, zf, image_index))
    return records


def _resolve_medxpert_image_refs(
    record: Mapping[str, Any],
    zf: zipfile.ZipFile,
    image_index: Mapping[str, str],
) -> dict[str, Any]:
    def resolve(value: Any) -> Any:
        if value is None or isinstance(value, (bytes, bytearray)) or hasattr(value, "save"):
            return value
        if isinstance(value, Mapping):
            return value
        name = Path(str(value)).name
        member = image_index.get(name)
        if member is None:
            for key, candidate in image_index.items():
                if key.lower() == name.lower():
                    member = candidate
                    break
        return zf.read(member) if member is not None else None

    out = dict(record)
    for key in ("image", "images", "image_list"):
        if key not in out:
            continue
        value = out[key]
        if isinstance(value, list):
            out[key] = [img for img in (resolve(item) for item in value) if img is not None]
        else:
            out[key] = resolve(value)
    return out


_MEDXPERT_HEADER_PATTERNS = [
    (
        "narrative",
        re.compile(
            r"^\s*(history|chief complaint|presentation|case|vignette|patient)\s*[:\-]",
            re.I | re.M,
        ),
    ),
    (
        "labs_exam",
        re.compile(
            r"^\s*(physical\s+exam|examination|exam findings|laboratory|labs?|"
            r"vital\s+signs?|vitals)\s*[:\-]",
            re.I | re.M,
        ),
    ),
    (
        "imaging",
        re.compile(
            r"^\s*(imaging|radiology|x[- ]?ray|ct scan|mri|ultrasound|pathology|"
            r"ecg|ekg)\s*[:\-]",
            re.I | re.M,
        ),
    ),
]


def _medxpert_question_text(row: Mapping[str, Any]) -> str:
    for field in ("question", "context", "case", "prompt", "input"):
        value = row.get(field)
        if value:
            return str(value)
    return ""


def _split_medxpert_headers(text: str) -> list[tuple[str | None, str]]:
    hits = []
    for category, pattern in _MEDXPERT_HEADER_PATTERNS:
        for match in pattern.finditer(text):
            hits.append((match.start(), match.end(), category))
    hits.sort()
    if not hits:
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return [(None, chunk) for chunk in chunks]

    out: list[tuple[str | None, str]] = []
    if hits[0][0] > 0:
        lead = text[: hits[0][0]].strip()
        if lead:
            out.append((None, lead))
    for idx, (start, _end, category) in enumerate(hits):
        end = hits[idx + 1][0] if idx + 1 < len(hits) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            out.append((category, chunk))
    return out


def _pick_three_medxpert_chunks(chunks: list[tuple[str | None, str]]) -> list[str] | None:
    if len(chunks) < 3:
        return None
    wanted = ("narrative", "labs_exam", "imaging")
    picked = {}
    for category, text in chunks:
        if category in wanted and category not in picked:
            picked[category] = text
        if len(picked) == 3:
            return [picked["narrative"], picked["labs_exam"], picked["imaging"]]
    return [text for _category, text in chunks[:3]]


def _medxpert_images(row: Mapping[str, Any]) -> list[Any]:
    for field in ("image", "images", "image_list"):
        value = row.get(field)
        if value is None:
            continue
        return list(value) if isinstance(value, list) else [value]
    return []


def _medxpert_choices_and_answer(row: Mapping[str, Any]) -> tuple[list[str], int]:
    choices_value = row.get("options") or row.get("choices") or []
    if isinstance(choices_value, Mapping):
        keys = sorted(str(key) for key in choices_value)
        choices = [str(choices_value[key]).strip() for key in keys]
    else:
        choices = [str(choice).strip() for choice in list(choices_value)]
    if len(choices) < 2:
        raise ValueError("MedXpertQA row does not contain at least two choices")

    answer = row.get("answer")
    if answer is None:
        answer = row.get("label", row.get("correct_answer"))
    if isinstance(answer, int) and 0 <= answer < len(choices):
        return choices, answer
    text = str(answer).strip()
    if len(text) == 1 and text.isalpha():
        idx = ord(text.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices, idx
    for idx, choice in enumerate(choices):
        if choice == text:
            return choices, idx
    raise ValueError(f"cannot map MedXpertQA answer {answer!r} to choices")


def load_image_set_order_mantis_eval(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    n = target or 18
    ds = _load_dataset("TIGER-Lab/Mantis-Eval", split="test", streaming=streaming)
    raw = []
    for row in ds:
        if row.get("question_type") != "multi-choice":
            continue
        images = row.get("images") or []
        if len(images) < 3:
            continue
        raw.append(dict(row))
        if len(raw) >= n * 4:
            break
    random.Random(SEED + 5).shuffle(raw)

    out = []
    for row in raw:
        images = [_decode_image_field(image) for image in row.get("images", [])]
        images = [image for image in images if image is not None][:6]
        if len(images) < 3:
            continue
        choices = _strip_choice_prefixes(row.get("options") or [])
        if len(choices) < 2:
            continue
        gold = _letter_from_text(row.get("answer"))
        if gold is None:
            continue
        question = re.sub(r"\s*<image>\s*", " ", str(row.get("question", ""))).strip()
        out.append(
            _make_image_set_example(
                dataset=dataset,
                raw_id=str(row.get("id", len(out))),
                question=question,
                images=images,
                choices=choices,
                gold_letter=gold,
                metadata={"category": row.get("category", "")},
            )
        )
        if len(out) >= n:
            break
    return out


def load_image_set_order_medframeqa(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    n = target or 195
    ds = _load_dataset("SuhaoYu1020/MedFrameQA", split="test", streaming=streaming)
    raw = []
    for row in ds:
        raw.append(dict(row))
        if len(raw) >= n * 6:
            break
    random.Random(SEED + 6).shuffle(raw)

    out = []
    for row in raw:
        images = _medframe_images(row)[:6]
        if len(images) < 3:
            continue
        choices = _choices_from_options(row.get("options") or row.get("choices"))
        if len(choices) < 2:
            continue
        gold = _letter_from_text(row.get("correct_answer") or row.get("answer"))
        if gold is None:
            continue
        out.append(
            _make_image_set_example(
                dataset=dataset,
                raw_id=str(row.get("question_id", row.get("id", len(out)))),
                question=str(row.get("question", "")),
                images=images,
                choices=choices,
                gold_letter=gold,
            )
        )
        if len(out) >= n:
            break
    return out


def _strip_choice_prefixes(values: Sequence[Any]) -> list[str]:
    choices = []
    for value in values:
        text = str(value).strip()
        match = re.match(r"^\(?([A-Z])\)?[\.\):\s]\s*(.+)$", text)
        choices.append(match.group(2).strip() if match else text)
    return choices


def _choices_from_options(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(value[key]).strip() for key in sorted(value)]
    if isinstance(value, Sequence) and not isinstance(value, str):
        return _strip_choice_prefixes(value)
    return []


def _letter_from_text(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    match = re.match(r"^\(?([A-Z])", text)
    return match.group(1) if match else None


def _medframe_images(row: Mapping[str, Any]) -> list[Any]:
    images = []
    for slot in range(12):
        key = f"image_{slot}"
        if key in row:
            image = _decode_image_field(row[key])
            if image is not None:
                images.append(image)
    if images:
        return images
    for field in ("images", "frames", "image_list"):
        value = row.get(field)
        if not isinstance(value, list):
            continue
        images = [_decode_image_field(item) for item in value]
        return [image for image in images if image is not None]
    return []


def _make_mixed_modality_example(
    *,
    dataset: DatasetProfile,
    raw_id: str,
    question: str,
    components_raw: Sequence[Mapping[str, Any]],
    images: Mapping[int, Any],
    gold: str,
    metadata: Mapping[str, Any] | None = None,
):
    runtime_example = _runtime_example_cls()
    components = []
    content: dict[str, Any] = {}
    for idx, raw in enumerate(components_raw):
        kind = str(raw.get("kind", "text"))
        component_kind = "image" if kind == "image" else "table" if kind == "table" else "text"
        ref = content_ref(dataset.name, raw_id, component_kind, idx)
        components.append(
            Component(
                component_id=f"{component_kind}_{idx}",
                kind=component_kind,
                content_ref=ref,
                label=None,
            )
        )
        if component_kind == "image":
            slot = int(raw.get("slot_idx", idx))
            image = _decode_image_field(images.get(slot))
            content[ref] = image if image is not None else "[image unavailable]"
        else:
            content[ref] = str(raw.get("text", "")).strip()
    if len(components) < 2:
        return None

    question_ref = content_ref(dataset.name, raw_id, "question")
    content[question_ref] = question
    item = AuditItem(
        item_id=f"{dataset.name}::{raw_id}",
        dataset=dataset.name,
        components=tuple(components),
        question_ref=question_ref,
        gold=gold,
        metadata=dict(metadata or {}),
    )
    return runtime_example(
        facet="mixed_modality_order",
        item=item,
        question=question,
        content=content,
        score_kind="exact_match",
        gold_normalized=normalize_answer("exact_match", gold),
        system_instruction=MIXED_MODALITY_SYSTEM,
        answer_instruction=MIXED_MODALITY_ANSWER_INSTRUCTION,
    )


def load_mixed_modality_order_mramg(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    del streaming
    n = target or 197
    qa_path = _hf_download("MRAMG/MRAMG-Bench", "recipe_mqa.jsonl")
    doc_path = _hf_download("MRAMG/MRAMG-Bench", "doc_recipe.jsonl")
    img_zip_path = _hf_download("MRAMG/MRAMG-Bench", "IMAGE.zip")

    docs: dict[int, str] = {}
    with open(doc_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if isinstance(entry, list) and len(entry) >= 2:
                docs[int(entry[0])] = str(entry[1])
            elif isinstance(entry, Mapping) and "id" in entry and "content" in entry:
                docs[int(entry["id"])] = str(entry["content"])

    candidates = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            provenance = row.get("provenance") or []
            if not provenance:
                continue
            doc_id = int(provenance[0])
            components = _parse_recipe_doc(docs.get(doc_id, ""))
            n_text = sum(1 for comp in components if comp["kind"] == "text")
            n_image = sum(1 for comp in components if comp["kind"] == "image")
            if n_text >= 3 and n_image >= 3:
                candidates.append((row, components, doc_id))
    random.Random(SEED).shuffle(candidates)

    out = []
    with zipfile.ZipFile(img_zip_path) as zf:
        for row, components, doc_id in candidates:
            images_list = list(row.get("images_list") or [])
            needed = sum(1 for comp in components if comp["kind"] == "image")
            if not images_list:
                continue
            while len(images_list) < needed:
                images_list.append(images_list[-1])
            loaded = {}
            missing = False
            for comp in components:
                if comp["kind"] != "image":
                    continue
                slot = int(comp["slot_idx"])
                try:
                    loaded[slot] = zf.read(f"IMAGE/images/RECIPE/{images_list[slot]}.jpg")
                except KeyError:
                    missing = True
                    break
            if missing:
                continue
            example = _make_mixed_modality_example(
                dataset=dataset,
                raw_id=str(row.get("id", f"recipe::{doc_id}")),
                question=str(row.get("question", "")),
                components_raw=components,
                images=loaded,
                gold=str(row.get("ground_truth", "")),
                metadata={"doc_id": doc_id},
            )
            if example is not None:
                out.append(example)
            if len(out) >= n:
                break
    return out


def load_mixed_modality_order_mmdocrag(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    del streaming
    n = target or 200
    qa_path = _hf_download("MMDocIR/MMDocRAG", "dev_15.jsonl")
    img_zip_path = _hf_download("MMDocIR/MMDocRAG", "images.zip")

    candidates = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text_quotes = row.get("text_quotes") or []
            image_quotes = row.get("img_quotes") or []
            gold_quotes = row.get("gold_quotes") or []
            has_text_gold = any(str(gold).startswith("text") for gold in gold_quotes)
            has_image_gold = any(str(gold).startswith("image") for gold in gold_quotes)
            if (
                len(text_quotes) >= 3
                and len(image_quotes) >= 3
                and has_text_gold
                and has_image_gold
            ):
                candidates.append(row)
    random.Random(SEED).shuffle(candidates)

    out = []
    with zipfile.ZipFile(img_zip_path) as zf:
        for row in candidates:
            components = [
                {"kind": "text", "text": quote.get("text", "")}
                for quote in row.get("text_quotes", [])
            ]
            loaded = {}
            for quote in row.get("img_quotes", []):
                img_path = quote.get("img_path") or quote.get("path")
                if not img_path:
                    continue
                try:
                    raw = zf.read(img_path)
                except KeyError:
                    continue
                slot_idx = len(loaded)
                loaded[slot_idx] = raw
                components.append({"kind": "image", "slot_idx": slot_idx})
            if len(loaded) < 3:
                continue
            gold = str(row.get("answer_short") or "").strip()
            if not gold:
                interleaved = str(row.get("answer_interleaved") or "")
                gold = interleaved.split(".")[0][:200] if interleaved else ""
            example = _make_mixed_modality_example(
                dataset=dataset,
                raw_id=str(row.get("doc_id", row.get("id", len(out)))),
                question=str(row.get("question", "")),
                components_raw=components,
                images=loaded,
                gold=gold,
            )
            if example is not None:
                out.append(example)
            if len(out) >= n:
                break
    return out


def load_mixed_modality_order_mmqa(
    dataset: DatasetProfile,
    target: int | None,
    streaming: bool = True,
) -> list[Any]:
    del streaming
    n = target or 200
    qa_path = _hf_download("TableQAKit/MMQA", "MMQA_dev.jsonl.gz")
    text_path = _hf_download("TableQAKit/MMQA", "MMQA_texts.jsonl.gz")
    image_path = _hf_download("TableQAKit/MMQA", "MMQA_images.jsonl.gz")

    texts = {}
    with gzip.open(text_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts[row.get("id") or row.get("doc_id")] = row.get("text") or row.get("body") or ""
    image_meta = {}
    with gzip.open(image_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            image_meta[row.get("id") or row.get("doc_id")] = row

    image_zip = _ensure_mmqa_image_zip(Path(qa_path).parent)
    if image_zip is None or not image_zip.exists():
        return []
    image_zf = zipfile.ZipFile(image_zip)

    candidates = []
    with gzip.open(qa_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            text_ids = metadata.get("text_doc_ids") or []
            image_ids = metadata.get("image_doc_ids") or []
            if len(text_ids) >= 3 and len(image_ids) >= 3:
                candidates.append(row)
    random.Random(SEED).shuffle(candidates)

    out = []
    try:
        for row in candidates:
            metadata = row.get("metadata") or {}
            text_ids = (metadata.get("text_doc_ids") or [])[:4]
            image_ids = (metadata.get("image_doc_ids") or [])[:4]
            components = []
            for text_id in text_ids:
                text = str(texts.get(text_id, ""))[:500]
                if text:
                    components.append({"kind": "text", "text": text})
            loaded = {}
            for image_id in image_ids:
                path = image_meta.get(image_id, {}).get("path") or f"{image_id}.jpg"
                raw = None
                for candidate in (path, f"final_dataset_images/{path}", f"images/{path}"):
                    try:
                        raw = image_zf.read(candidate)
                        break
                    except KeyError:
                        continue
                if raw is None:
                    continue
                slot_idx = len(loaded)
                loaded[slot_idx] = raw
                components.append({"kind": "image", "slot_idx": slot_idx})
            if len(loaded) < 3:
                continue
            answers = row.get("answers") or []
            gold = str(answers[0].get("answer", "") if answers else "")
            example = _make_mixed_modality_example(
                dataset=dataset,
                raw_id=str(row.get("qid", row.get("id", len(out)))),
                question=str(row.get("question", "")),
                components_raw=components,
                images=loaded,
                gold=gold,
            )
            if example is not None:
                out.append(example)
            if len(out) >= n:
                break
    finally:
        image_zf.close()
    return out


def _parse_recipe_doc(doc_text: str) -> list[dict[str, Any]]:
    parts = doc_text.split("<PIC>")
    out = []
    slot = 0
    for idx, part in enumerate(parts):
        text = part.strip()
        if text:
            out.append({"kind": "text", "text": text})
        if idx < len(parts) - 1:
            out.append({"kind": "image", "slot_idx": slot})
            slot += 1
    return out


def _hf_download(repo: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download  # type: ignore

    return Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=filename))


def _ensure_mmqa_image_zip(cache_dir: Path) -> Path | None:
    zip_path = cache_dir / "mmqa_images.zip"
    if zip_path.exists():
        return zip_path
    try:
        import urllib.request

        url = (
            "https://multimodalqa-images.s3-us-west-2.amazonaws.com/"
            "final_dataset_images/final_dataset_images.zip"
        )
        urllib.request.urlretrieve(url, str(zip_path))
        return zip_path
    except Exception:
        return None


_LOADERS: dict[tuple[str, str], Loader] = {
    ("option_order", "mmlu_pro"): load_option_order_mmlu_pro,
    ("option_order", "commonsenseqa"): load_option_order_commonsenseqa,
    ("option_order", "mathvision"): load_option_order_mathvision,
    ("evidence_chunk_order", "hotpotqa"): load_evidence_chunk_order_hotpotqa,
    ("evidence_chunk_order", "musique"): load_evidence_chunk_order_musique,
    ("evidence_chunk_order", "medxpertqa"): load_evidence_chunk_order_medxpertqa,
    ("document_rank_order", "multihop_rag"): load_document_rank_order_multihop_rag,
    ("document_rank_order", "hotpotqa"): load_document_rank_order_hotpotqa,
    ("image_set_order", "mantis_eval"): load_image_set_order_mantis_eval,
    ("image_set_order", "medframeqa"): load_image_set_order_medframeqa,
    ("mixed_modality_order", "mramg"): load_mixed_modality_order_mramg,
    ("mixed_modality_order", "mmdocrag"): load_mixed_modality_order_mmdocrag,
    ("mixed_modality_order", "mmqa"): load_mixed_modality_order_mmqa,
}
