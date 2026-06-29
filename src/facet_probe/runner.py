"""Execution helpers for profile-backed Facet-Probe runs."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from facet_probe.facets import get_facet
from facet_probe.manifests import trial_manifest_rows
from facet_probe.metrics import audit_records, read_jsonl, summarize_groups, write_csv, write_json
from facet_probe.profiles import DatasetProfile, EvaluationProfile, ModelProfile
from facet_probe.prompts import SYSTEM_INSTRUCTION, build_mcq_question_block
from facet_probe.reports import write_evaluation_report
from facet_probe.schema import AuditItem, Component
from facet_probe.scoring import normalize_answer, score_answer
from facet_probe.templates import content_ref, render_ordered_text_prompt

QUESTION_FIELDS = ("question", "query", "prompt", "problem", "input", "case")
CHOICE_FIELDS = ("choices", "options", "answer_choices", "choice", "candidates")
ANSWER_FIELDS = (
    "answer",
    "answerKey",
    "answer_key",
    "label",
    "gold",
    "gold_answer",
    "correct_answer",
    "answer_index",
    "answer_idx",
    "target",
)
EVIDENCE_FIELDS = (
    "evidence",
    "evidence_list",
    "context",
    "contexts",
    "paragraphs",
    "passages",
    "supporting_facts",
)
DOCUMENT_FIELDS = (
    "documents",
    "document",
    "retrieved_documents",
    "retrievals",
    "ranked_documents",
    "passages",
    "context",
)
IMAGE_FIELDS = ("images", "image_list", "image", "frames")
TABLE_FIELDS = ("tables", "table", "metadata_table")
TEXT_KEYS = ("text", "content", "passage", "paragraph_text", "body", "sentence", "value")
DEFAULT_ANSWER_INSTRUCTION = (
    "Answer with the final answer only. For multiple-choice questions, "
    "use the answer letter."
)


@dataclass(frozen=True)
class RuntimeExample:
    """A normalized item plus runtime-only content needed to render prompts."""

    facet: str
    item: AuditItem
    question: str | None
    content: dict[str, Any]
    score_kind: str
    gold_normalized: str | None
    fixed_components: tuple[Component, ...] = ()
    system_instruction: str = SYSTEM_INSTRUCTION
    answer_instruction: str = DEFAULT_ANSWER_INSTRUCTION


@dataclass(frozen=True)
class PromptTextPiece:
    """One text block in the rendered model input."""

    text: str
    label: str | None = None

    def render(self) -> str:
        if self.label:
            return f"{self.label}: {self.text.strip()}"
        return self.text.strip()


@dataclass(frozen=True)
class PromptImagePiece:
    """One image block in the rendered model input."""

    image: Any
    label: str | None = None

    def render_text_only(self) -> str:
        return f"[{self.label or 'image'}]"


PromptPiece = PromptTextPiece | PromptImagePiece


@dataclass(frozen=True)
class PromptBundle:
    """Both text-only and multimodal renderings of one trial input."""

    text: str
    pieces: tuple[PromptPiece, ...]


@dataclass(frozen=True)
class ModelResponse:
    """Text returned by a model adapter."""

    text: str
    raw: str | None = None
    metadata: dict[str, Any] | None = None


class ModelAdapter(Protocol):
    """Small adapter protocol used by the public runner."""

    def generate(
        self,
        prompt: str,
        *,
        pieces: Sequence[PromptPiece],
        example: RuntimeExample,
        manifest_row: Mapping[str, Any],
        max_new_tokens: int,
    ) -> ModelResponse:
        """Return one model response for one ordered prompt."""

    def close(self) -> None:
        """Release adapter resources."""


class RowUnsupported(ValueError):
    """Raised when a generic row adapter cannot normalize a dataset row."""


def execute_profile(
    profile: EvaluationProfile,
    output_dir: str | Path,
    *,
    items_jsonl: str | Path | None = None,
    item_facet: str = "option_order",
    limit_items: int | None = None,
    limit_trials: int | None = None,
    streaming: bool = True,
    include_raw_outputs: bool = True,
    max_new_tokens: int | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Run a profile and write manifest/trial/report artifacts.

    The execution path is intentionally adapter-neutral: dataset rows are
    normalized into ``AuditItem`` objects, deterministic orderings are emitted
    to ``manifest.jsonl``, model outputs are normalized into ``trials.jsonl``,
    and report artifacts are built from those trial records.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "manifest": output / "manifest.jsonl",
        "trials": output / "trials.jsonl",
        "summary": output / "summary.json",
        "group_summary": output / "group_summary.csv",
        "run_status": output / "run_status.json",
        "report_dir": output / "report",
    }

    examples, skipped = load_runtime_examples(
        profile,
        items_jsonl=items_jsonl,
        item_facet=item_facet,
        limit_items=limit_items,
        streaming=streaming,
    )
    if not examples:
        raise RuntimeError("no runnable examples were loaded for this profile")
    if skipped["shortfalls"] and not allow_partial:
        raise RuntimeError(
            "paper-run did not load the requested audited item counts. "
            f"Shortfalls: {skipped['shortfalls'][:5]}. "
            "Use --limit-items for a smoke run or --allow-partial for development."
        )

    manifest_rows, by_key = _build_manifest(profile, examples, limit_trials=limit_trials)
    _write_jsonl(files["manifest"], manifest_rows)

    existing_records = read_jsonl(files["trials"]) if files["trials"].exists() else []
    completed = _completed_keys(existing_records)
    report_records = list(existing_records)
    max_tokens = max_new_tokens or int(
        profile.generation_defaults.get("max_new_tokens_default", 64)
    )
    adapter_errors: list[dict[str, str]] = []

    for model in profile.models:
        adapter = create_model_adapter(model)
        try:
            for manifest_row in manifest_rows:
                key = _trial_key(model.name, manifest_row)
                if key in completed:
                    continue
                example = by_key[(str(manifest_row["facet"]), str(manifest_row["item_id"]))]
                record = _execute_trial(
                    adapter=adapter,
                    model=model,
                    example=example,
                    manifest_row=manifest_row,
                    max_new_tokens=max_tokens,
                    include_raw_outputs=include_raw_outputs,
                )
                _append_jsonl(files["trials"], record)
                report_records.append(record)
                completed.add(key)
        except Exception as exc:
            adapter_errors.append({"model": model.name, "error": str(exc)})
            raise
        finally:
            adapter.close()

    summary = asdict(audit_records(report_records, label=profile.name))
    group_rows = summarize_groups(report_records)
    write_json(files["summary"], summary)
    write_csv(files["group_summary"], group_rows)
    report_files = write_evaluation_report(files["report_dir"], report_records, label=profile.name)

    status = {
        "status": "completed",
        "profile": profile.name,
        "n_models": len(profile.models),
        "n_examples": len(examples),
        "n_manifest_rows": len(manifest_rows),
        "n_trial_records": len(report_records),
        "skipped_row_counts": dict(skipped["counts"]),
        "skipped_row_examples": skipped["examples"],
        "shortfalls": skipped["shortfalls"],
        "adapter_errors": adapter_errors,
        "files": {
            "manifest": str(files["manifest"]),
            "trials": str(files["trials"]),
            "summary": str(files["summary"]),
            "group_summary": str(files["group_summary"]),
            "run_status": str(files["run_status"]),
            "report": {name: str(path) for name, path in report_files.items()},
        },
    }
    write_json(files["run_status"], status)
    return status


def load_runtime_examples(
    profile: EvaluationProfile,
    *,
    items_jsonl: str | Path | None = None,
    item_facet: str = "option_order",
    limit_items: int | None = None,
    streaming: bool = True,
) -> tuple[list[RuntimeExample], dict[str, Any]]:
    """Load normalized runtime examples from JSONL or configured HF datasets."""

    if items_jsonl is not None:
        examples = [
            _example_from_audit_item(AuditItem.from_mapping(row), facet=item_facet)
            for row in read_jsonl(items_jsonl)
        ]
        if limit_items is not None:
            examples = examples[:limit_items]
        return examples, {"counts": Counter(), "examples": [], "shortfalls": []}

    examples: list[RuntimeExample] = []
    skipped_counts: Counter[str] = Counter()
    skipped_examples: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    for dataset in profile.datasets:
        target = limit_items if limit_items is not None else _audited_n_int(dataset.audited_n)
        counts_by_facet: Counter[str] = Counter()
        paper_examples = _load_paper_dataset_runtime_examples(
            dataset,
            target=target,
            streaming=streaming,
        )
        if paper_examples is not None:
            examples.extend(paper_examples)
            counts_by_facet.update(example.facet for example in paper_examples)
            if target is not None:
                for facet in dataset.facets:
                    loaded = counts_by_facet[facet]
                    if loaded < target:
                        shortfalls.append(
                            {
                                "dataset": dataset.name,
                                "facet": facet,
                                "expected": target,
                                "loaded": loaded,
                            }
                        )
            continue
        for row_idx, row in enumerate(_iter_hf_dataset_rows(dataset, streaming=streaming)):
            pending_facets = [
                facet
                for facet in dataset.facets
                if target is None or counts_by_facet[facet] < target
            ]
            if not pending_facets:
                break
            row_map = dict(row)
            for facet in pending_facets:
                try:
                    examples.append(_example_from_hf_row(row_map, dataset, facet, row_idx))
                    counts_by_facet[facet] += 1
                except RowUnsupported as exc:
                    reason = f"{dataset.name}:{facet}:{exc}"
                    skipped_counts[reason] += 1
                    if len(skipped_examples) < 50:
                        skipped_examples.append(
                            {
                                "dataset": dataset.name,
                                "facet": facet,
                                "row_idx": row_idx,
                                "reason": str(exc),
                            }
                        )
        if target is not None:
            for facet in dataset.facets:
                loaded = counts_by_facet[facet]
                if loaded < target:
                    shortfalls.append(
                        {
                            "dataset": dataset.name,
                            "facet": facet,
                            "expected": target,
                            "loaded": loaded,
                        }
                    )
    return examples, {
        "counts": skipped_counts,
        "examples": skipped_examples,
        "shortfalls": shortfalls,
    }


def _load_paper_dataset_runtime_examples(
    dataset: DatasetProfile,
    *,
    target: int | None,
    streaming: bool,
) -> list[RuntimeExample] | None:
    try:
        from facet_probe.paper_loaders import load_paper_dataset_examples
    except ImportError:
        return None
    loaded = load_paper_dataset_examples(dataset, target=target, streaming=streaming)
    return None if loaded is None else list(loaded)


def create_model_adapter(model: ModelProfile) -> ModelAdapter:
    """Instantiate a model adapter for the public runner."""

    if model.provider == "mock":
        return MockAdapter(model)
    if model.provider == "huggingface":
        return HuggingFaceLocalAdapter(model)
    if model.provider in {"openai", "openai-compatible"}:
        return OpenAIChatAdapter(model)
    if model.provider == "anthropic":
        return AnthropicChatAdapter(model)
    if model.provider == "google":
        return GoogleGenAIAdapter(model)
    raise RuntimeError(f"no public runner adapter for provider {model.provider!r}")


class MockAdapter:
    """Deterministic adapter for CI, examples, and local artifact smoke tests."""

    def __init__(self, model: ModelProfile):
        self.model = model

    def generate(
        self,
        prompt: str,
        *,
        pieces: Sequence[PromptPiece],
        example: RuntimeExample,
        manifest_row: Mapping[str, Any],
        max_new_tokens: int,
    ) -> ModelResponse:
        del prompt, pieces, max_new_tokens
        if example.score_kind == "option_content_idx":
            answer = _display_letter_for_source_index(
                example.gold_normalized,
                manifest_row.get("permutation") or (),
            )
            return ModelResponse(text=f"Answer: {answer or 'A'}", raw=f"Answer: {answer or 'A'}")
        if example.score_kind == "mcq_letter":
            answer = example.gold_normalized or "A"
            return ModelResponse(text=f"Answer: {answer}", raw=f"Answer: {answer}")
        if example.score_kind == "llm_judge_gold_match":
            return ModelResponse(text="true", raw="true")
        return ModelResponse(
            text=example.gold_normalized or "mock-answer",
            raw=example.gold_normalized or "mock-answer",
        )

    def close(self) -> None:
        return None


class HuggingFaceLocalAdapter:
    """Local HuggingFace adapter with Qwen-VL and text-generation paths."""

    def __init__(self, model: ModelProfile):
        self.model = model
        self._pipe = None
        self._processor = None
        self._model = None
        self._device = None

    def _is_qwen_vl(self) -> bool:
        repo = (self.model.hf_repo or self.model.model_id).lower()
        return "qwen" in repo and ("vl" in repo or "qwen3.5" in repo or "qwen3-5" in repo)

    def _load_text_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "Install local model support with `pip install -e '.[models]'` "
                "or run `facet-probe paper-run --prepare-only` to create run manifests."
            ) from exc

        repo = self.model.hf_repo or self.model.model_id
        try:
            self._pipe = pipeline(
                "text-generation",
                model=repo,
                device_map=self.model.generation.get("device_map", "auto"),
                trust_remote_code=bool(self.model.generation.get("trust_remote_code", True)),
            )
        except Exception as exc:  # pragma: no cover - model/hardware dependent
            raise RuntimeError(
                f"could not load HuggingFace text-generation model {repo!r}: {exc}. "
                "Vision-language models may need a custom adapter; use "
                "`--prepare-only` to write the reproducible run profile without inference."
            ) from exc
        return self._pipe

    def _load_qwen_vl(self):
        if self._model is not None and self._processor is not None:
            return self._processor, self._model
        try:
            import torch  # type: ignore
            from transformers import AutoModelForImageTextToText as AutoVLM  # type: ignore
            from transformers import AutoProcessor  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "Install local VLM support with `pip install -e '.[models]'`. "
                "Qwen-family multimodal models require transformers, torch, and qwen-vl-utils."
            ) from exc

        repo = self.model.hf_repo or self.model.model_id
        self._processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        dtype_name = str(self.model.generation.get("dtype", "bfloat16"))
        kwargs: dict[str, Any] = {
            "device_map": self.model.generation.get("device_map", "auto"),
            "trust_remote_code": True,
        }
        attn_impl = self.model.generation.get("attn_implementation")
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        if self.model.generation.get("load_in_4bit"):
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except ImportError as exc:  # pragma: no cover - depends on optional extras
                raise RuntimeError(
                    "4-bit loading requires bitsandbytes support in transformers."
                ) from exc
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["dtype"] = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(dtype_name, torch.bfloat16)
        self._model = AutoVLM.from_pretrained(repo, **kwargs).eval()
        self._device = getattr(self._model, "device", None)
        return self._processor, self._model

    def generate(
        self,
        prompt: str,
        *,
        pieces: Sequence[PromptPiece],
        example: RuntimeExample,
        manifest_row: Mapping[str, Any],
        max_new_tokens: int,
    ) -> ModelResponse:
        del manifest_row
        if self._is_qwen_vl():
            return self._generate_qwen_vl(
                pieces,
                max_new_tokens=max_new_tokens,
                system_instruction=example.system_instruction,
            )
        pipe = self._load_text_pipeline()
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=bool(self.model.generation.get("do_sample", False)),
            return_full_text=False,
        )
        text = _extract_pipeline_text(outputs)
        return ModelResponse(text=text, raw=text)

    def _generate_qwen_vl(
        self,
        pieces: Sequence[PromptPiece],
        *,
        max_new_tokens: int,
        system_instruction: str,
    ) -> ModelResponse:
        processor, model = self._load_qwen_vl()
        try:
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover - guarded by _load_qwen_vl
            raise RuntimeError("torch is required for local HuggingFace VLM inference") from exc

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
            {"role": "user", "content": _pieces_to_hf_content(pieces)},
        ]
        enable_thinking = bool(self.model.generation.get("enable_thinking", False))
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        image_inputs = [piece.image for piece in pieces if isinstance(piece, PromptImagePiece)]
        video_inputs = None
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore

            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            image_inputs = image_inputs or None

        kwargs: dict[str, Any] = {"text": [text], "padding": True, "return_tensors": "pt"}
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs
        inputs = processor(**kwargs)
        device = self._device
        if device is not None:
            inputs = inputs.to(device)

        sampling = {
            "do_sample": bool(self.model.generation.get("do_sample", False)),
            "temperature": self.model.generation.get("temperature"),
            "top_p": self.model.generation.get("top_p"),
        }
        sampling = {key: value for key, value in sampling.items() if value is not None}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, **sampling)
        new_tokens = out[:, inputs.input_ids.shape[1] :]
        raw = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        answer, thinking = _split_thinking(raw) if enable_thinking else (raw.strip(), None)
        metadata = {"thinking": thinking} if thinking else None
        return ModelResponse(text=answer, raw=raw, metadata=metadata)

    def close(self) -> None:
        self._pipe = None
        self._processor = None
        self._model = None


class OpenAIChatAdapter:
    """OpenAI and OpenAI-compatible chat-completions adapter."""

    def __init__(self, model: ModelProfile):
        self.model = model
        self._client = None

    def _load(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "Install provider support with `pip install -e '.[providers]'`."
            ) from exc
        if self.model.provider == "openai-compatible":
            base_url = self.model.generation.get("endpoint_url") or os.environ.get(
                "FACET_PROBE_OPENAI_COMPAT_BASE_URL"
            )
            api_key = _first_env(self.model.env) or os.environ.get(
                "FACET_PROBE_OPENAI_COMPAT_API_KEY",
                "EMPTY",
            )
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            self._client = OpenAI(api_key=_first_env(self.model.env))
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        pieces: Sequence[PromptPiece],
        example: RuntimeExample,
        manifest_row: Mapping[str, Any],
        max_new_tokens: int,
    ) -> ModelResponse:
        del prompt, manifest_row
        client = self._load()
        if self.model.provider == "openai":
            response = client.responses.create(
                model=self.model.model_id,
                instructions=example.system_instruction,
                input=[
                    {
                        "role": "user",
                        "content": _pieces_to_openai_responses_content(pieces),
                    }
                ],
                max_output_tokens=max_new_tokens,
            )
            text = getattr(response, "output_text", "") or ""
        else:
            response = client.chat.completions.create(
                model=self.model.model_id,
                messages=[
                    {"role": "system", "content": example.system_instruction},
                    {"role": "user", "content": _pieces_to_openai_chat_content(pieces)},
                ],
                max_tokens=max_new_tokens,
                temperature=float(self.model.generation.get("temperature", 0)),
            )
            text = response.choices[0].message.content or ""
        return ModelResponse(text=text, raw=text)

    def close(self) -> None:
        self._client = None


class AnthropicChatAdapter:
    """Anthropic Messages API adapter."""

    def __init__(self, model: ModelProfile):
        self.model = model
        self._client = None

    def _load(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "Install provider support with `pip install -e '.[providers]'`."
            ) from exc
        self._client = Anthropic(api_key=_first_env(self.model.env))
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        pieces: Sequence[PromptPiece],
        example: RuntimeExample,
        manifest_row: Mapping[str, Any],
        max_new_tokens: int,
    ) -> ModelResponse:
        del prompt, manifest_row
        client = self._load()
        response = client.messages.create(
            model=self.model.model_id,
            max_tokens=max_new_tokens,
            temperature=float(self.model.generation.get("temperature", 0)),
            system=example.system_instruction,
            messages=[{"role": "user", "content": _pieces_to_anthropic_content(pieces)}],
        )
        text = "".join(
            getattr(block, "text", "")
            for block in getattr(response, "content", [])
            if getattr(block, "type", "text") == "text"
        )
        return ModelResponse(text=text, raw=text)

    def close(self) -> None:
        self._client = None


class GoogleGenAIAdapter:
    """Google GenAI adapter for Gemini-family models."""

    def __init__(self, model: ModelProfile):
        self.model = model
        self._client = None

    def _load(self):
        if self._client is not None:
            return self._client
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "Install provider support with `pip install -e '.[providers]'`."
            ) from exc
        api_key = _first_env(self.model.env)
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        return self._client

    def generate(
        self,
        prompt: str,
        *,
        pieces: Sequence[PromptPiece],
        example: RuntimeExample,
        manifest_row: Mapping[str, Any],
        max_new_tokens: int,
    ) -> ModelResponse:
        del prompt, manifest_row
        client = self._load()
        config = {
            "temperature": float(self.model.generation.get("temperature", 0)),
            "max_output_tokens": max_new_tokens,
            "system_instruction": example.system_instruction,
        }
        from google.genai import types  # type: ignore

        response = client.models.generate_content(
            model=self.model.model_id,
            contents=[
                types.Content(role="user", parts=_pieces_to_google_parts(pieces))
            ],
            config=config,
        )
        text = getattr(response, "text", "") or ""
        return ModelResponse(text=text, raw=text)

    def close(self) -> None:
        self._client = None


def _execute_trial(
    *,
    adapter: ModelAdapter,
    model: ModelProfile,
    example: RuntimeExample,
    manifest_row: Mapping[str, Any],
    max_new_tokens: int,
    include_raw_outputs: bool,
) -> dict[str, Any]:
    bundle = _build_prompt_bundle(example, manifest_row)
    t0 = time.time()
    response = adapter.generate(
        bundle.text,
        pieces=bundle.pieces,
        example=example,
        manifest_row=manifest_row,
        max_new_tokens=max_new_tokens,
    )
    latency = time.time() - t0
    answer_normalized = normalize_answer(
        example.score_kind,
        response.text,
        n_choices=len(example.item.choices),
        permutation=manifest_row.get("permutation") or (),
    )
    correct = score_answer(answer_normalized, example.gold_normalized)
    record: dict[str, Any] = {
        "schema_version": 1,
        "facet": example.facet,
        "dataset": example.item.dataset,
        "model": model.name,
        "provider": model.provider,
        "model_id": model.model_id,
        "item_id": example.item.item_id,
        "ordering_idx": int(manifest_row["ordering_idx"]),
        "permutation": list(manifest_row["permutation"]),
        "ordered_component_ids": list(manifest_row["ordered_component_ids"]),
        "answer_normalized": answer_normalized,
        "gold_normalized": example.gold_normalized,
        "correct": correct,
        "score_kind": example.score_kind,
        "n_choices": len(example.item.choices),
        "prompt_hash": _prompt_hash(bundle),
        "latency_sec": round(latency, 3),
    }
    if include_raw_outputs:
        record["raw_output"] = response.raw if response.raw is not None else response.text
    if example.facet == "mixed_modality_order" and example.question:
        record["question"] = example.question
    if response.metadata:
        record["adapter_metadata"] = response.metadata
    return record


def _build_manifest(
    profile: EvaluationProfile,
    examples: Sequence[RuntimeExample],
    *,
    limit_trials: int | None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], RuntimeExample]]:
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], RuntimeExample] = {}
    for example in examples:
        by_key[(example.facet, example.item.item_id)] = example
        for row in trial_manifest_rows(
            [example.item],
            facet=example.facet,
            k=profile.k_orderings,
            seed=profile.seed,
        ):
            row["score_kind"] = example.score_kind
            row["gold_normalized"] = example.gold_normalized
            row["n_choices"] = len(example.item.choices)
            rows.append(row)
    if limit_trials is not None:
        rows = rows[:limit_trials]
    return rows, by_key


def _build_prompt_bundle(
    example: RuntimeExample,
    manifest_row: Mapping[str, Any],
) -> PromptBundle:
    instruction = example.answer_instruction
    ordered_ids = list(manifest_row["ordered_component_ids"])
    text = render_ordered_text_prompt(
        example.item,
        ordered_ids,
        resolve_content=lambda component: _describe_value(
            example.content.get(component.content_ref, component.content_ref)
        ),
        question=example.question,
        instruction=instruction,
    )
    if example.fixed_components:
        fixed_lines = [
            _render_component_text(component, example.content)
            for component in example.fixed_components
        ]
        text = "\n".join(["Fixed context:", *fixed_lines, text])

    by_id = {component.component_id: component for component in example.item.components}
    component_index = {
        component.component_id: idx
        for idx, component in enumerate(example.item.components)
    }
    ordered = [by_id[component_id] for component_id in ordered_ids]
    if example.question and example.item.choices and all(
        component.kind == "choice" for component in ordered
    ):
        ordered_choices = [
            example.item.choices[component_index[component.component_id]]
            for component in ordered
        ]
        pieces: list[PromptPiece] = [
            _component_to_piece(component, example.content)
            for component in example.fixed_components
        ]
        pieces.append(
            PromptTextPiece(
                text="\n".join(
                    [build_mcq_question_block(example.question, ordered_choices), instruction]
                )
            )
        )
        return PromptBundle(
            text=text,
            pieces=tuple(pieces),
        )

    pieces: list[PromptPiece] = [
        _component_to_piece(component, example.content)
        for component in example.fixed_components
    ]
    for component in ordered:
        pieces.append(_component_to_piece(component, example.content))

    if example.question:
        if example.item.choices:
            question_block = build_mcq_question_block(example.question, example.item.choices)
        else:
            question_block = "Question: " + example.question.strip()
        pieces.append(PromptTextPiece(text="\n".join([question_block, instruction])))
    return PromptBundle(text=text, pieces=tuple(pieces))


def _component_to_piece(component: Component, content: Mapping[str, Any]) -> PromptPiece:
    value = content.get(component.content_ref, component.content_ref)
    if component.kind == "image":
        image = _coerce_pil_image(value)
        if image is not None:
            return PromptImagePiece(image=image, label=component.label)
    return PromptTextPiece(text=_describe_value(value), label=component.label)


def _render_component_text(component: Component, content: Mapping[str, Any]) -> str:
    value = content.get(component.content_ref, component.content_ref)
    label = component.label or component.component_id
    return f"{label}: {_describe_value(value)}"


def _example_from_audit_item(item: AuditItem, *, facet: str) -> RuntimeExample:
    score_kind = get_facet(facet).score_kind
    content = _content_from_audit_item(item)
    return RuntimeExample(
        facet=facet,
        item=item,
        question=str(item.question_ref) if item.question_ref else None,
        content=content,
        score_kind=score_kind,
        gold_normalized=_gold_from_item(item, score_kind),
    )


def _example_from_hf_row(
    row: Mapping[str, Any],
    dataset: DatasetProfile,
    facet: str,
    row_idx: int,
) -> RuntimeExample:
    if facet == "option_order":
        return _mcq_example(row, dataset, facet, row_idx)
    if facet == "evidence_chunk_order":
        return _sequence_example(row, dataset, facet, row_idx, EVIDENCE_FIELDS, "text")
    if facet == "document_rank_order":
        return _sequence_example(row, dataset, facet, row_idx, DOCUMENT_FIELDS, "document")
    if facet == "image_set_order":
        return _image_example(row, dataset, facet, row_idx)
    if facet == "mixed_modality_order":
        return _mixed_example(row, dataset, facet, row_idx)
    raise RowUnsupported(f"facet {facet!r} is not supported by the generic runner")


def _fixed_image_components(
    row: Mapping[str, Any],
    *,
    dataset: str,
    item_id: str,
) -> tuple[Component, ...]:
    for field in IMAGE_FIELDS:
        if field not in row or row[field] is None:
            continue
        images = _as_sequence(row[field])
        return tuple(
            Component(
                component_id=f"fixed_image_{idx}",
                kind="image",
                content_ref=content_ref(dataset, item_id, field, idx),
                label=f"Image {idx + 1}",
                metadata={"source_field": field, "source_index": idx, "fixed": True},
            )
            for idx in range(len(images))
        )
    return ()


def _mcq_example(
    row: Mapping[str, Any],
    dataset: DatasetProfile,
    facet: str,
    row_idx: int,
) -> RuntimeExample:
    raw_id = _row_id(row, row_idx)
    question = _first_text(row, QUESTION_FIELDS)
    choices_field, choices_value = _first_present(row, CHOICE_FIELDS)
    choices = _text_sequence(choices_value)
    min_choices = int(dataset.filters.get("min_choices", 2))
    if len(choices) < min_choices:
        raise RowUnsupported(f"expected at least {min_choices} choices")
    _answer_field, answer = _first_present(row, ANSWER_FIELDS)
    gold_idx = _answer_to_index(answer, choices)
    components = tuple(
        Component(
            component_id=f"choice_{idx}",
            kind="choice",
            content_ref=content_ref(dataset.name, raw_id, choices_field, idx),
            label=chr(ord("A") + idx),
        )
        for idx in range(len(choices))
    )
    item = AuditItem(
        item_id=f"{dataset.name}::{raw_id}",
        dataset=dataset.name,
        components=components,
        question_ref=content_ref(dataset.name, raw_id, "question"),
        choices=tuple(choices),
        gold=str(gold_idx),
        metadata={"hf_repo": dataset.hf_repo, "split": dataset.split},
    )
    content = {item.question_ref or "": question}
    content.update({
        component.content_ref: choices[idx]
        for idx, component in enumerate(components)
    })
    fixed_components = _fixed_image_components(row, dataset=dataset.name, item_id=raw_id)
    for component in fixed_components:
        field = component.metadata.get("source_field")
        index = component.metadata.get("source_index")
        if field in row and index is not None:
            content[component.content_ref] = _as_sequence(row[field])[int(index)]
    return RuntimeExample(
        facet=facet,
        item=item,
        question=question,
        content=content,
        score_kind=get_facet(facet).score_kind,
        gold_normalized=str(gold_idx),
        fixed_components=fixed_components,
    )


def _sequence_example(
    row: Mapping[str, Any],
    dataset: DatasetProfile,
    facet: str,
    row_idx: int,
    fields: Sequence[str],
    kind: str,
) -> RuntimeExample:
    raw_id = _row_id(row, row_idx)
    question = _first_text(row, QUESTION_FIELDS)
    field, value = _first_present(row, fields)
    texts = _text_sequence(value)
    texts = [text for text in texts if text.strip()]
    if len(texts) < 2:
        raise RowUnsupported(f"expected at least 2 orderable {kind} components")
    answer = _optional_first_text(row, ANSWER_FIELDS)
    components = tuple(
        Component(
            component_id=f"{kind}_{idx}",
            kind=kind,
            content_ref=content_ref(dataset.name, raw_id, field, idx),
            label=f"{kind.title()} {idx + 1}",
        )
        for idx in range(len(texts))
    )
    item = AuditItem(
        item_id=f"{dataset.name}::{raw_id}",
        dataset=dataset.name,
        components=components,
        question_ref=content_ref(dataset.name, raw_id, "question"),
        gold=answer,
        metadata={"hf_repo": dataset.hf_repo, "split": dataset.split},
    )
    content = {item.question_ref or "": question}
    content.update({
        component.content_ref: texts[idx]
        for idx, component in enumerate(components)
    })
    score_kind = get_facet(facet).score_kind
    return RuntimeExample(
        facet=facet,
        item=item,
        question=question,
        content=content,
        score_kind=score_kind,
        gold_normalized=normalize_answer(score_kind, answer),
    )


def _image_example(
    row: Mapping[str, Any],
    dataset: DatasetProfile,
    facet: str,
    row_idx: int,
) -> RuntimeExample:
    raw_id = _row_id(row, row_idx)
    question = _first_text(row, QUESTION_FIELDS)
    field, value = _first_present(row, IMAGE_FIELDS)
    images = _as_sequence(value)
    min_images = int(dataset.filters.get("min_images", 2))
    if len(images) < min_images:
        raise RowUnsupported(f"expected at least {min_images} images")
    choices = _maybe_choices(row)
    answer = _optional_first(row, ANSWER_FIELDS)
    gold = _gold_letter(answer, choices)
    components = tuple(
        Component(
            component_id=f"image_{idx}",
            kind="image",
            content_ref=content_ref(dataset.name, raw_id, field, idx),
            label=f"Image {idx + 1}",
        )
        for idx in range(len(images))
    )
    item = AuditItem(
        item_id=f"{dataset.name}::{raw_id}",
        dataset=dataset.name,
        components=components,
        question_ref=content_ref(dataset.name, raw_id, "question"),
        choices=tuple(choices),
        gold=gold,
        metadata={"hf_repo": dataset.hf_repo, "split": dataset.split},
    )
    content = {item.question_ref or "": question}
    content.update({
        component.content_ref: images[idx]
        for idx, component in enumerate(components)
    })
    return RuntimeExample(
        facet=facet,
        item=item,
        question=question,
        content=content,
        score_kind=get_facet(facet).score_kind,
        gold_normalized=gold,
    )


def _mixed_example(
    row: Mapping[str, Any],
    dataset: DatasetProfile,
    facet: str,
    row_idx: int,
) -> RuntimeExample:
    raw_id = _row_id(row, row_idx)
    question = _first_text(row, QUESTION_FIELDS)
    components: list[Component] = []
    content: dict[str, str] = {}
    for field in (*EVIDENCE_FIELDS, *DOCUMENT_FIELDS):
        if field not in row or row[field] is None:
            continue
        for text in _text_sequence(row[field]):
            if not text.strip():
                continue
            ref = content_ref(dataset.name, raw_id, field, len(components))
            components.append(
                Component(
                    component_id=f"text_{len(components)}",
                    kind="text",
                    content_ref=ref,
                    label=f"Text {len(components) + 1}",
                )
            )
            content[ref] = text
    for field in IMAGE_FIELDS:
        if field not in row or row[field] is None:
            continue
        for image in _as_sequence(row[field]):
            ref = content_ref(dataset.name, raw_id, field, len(components))
            components.append(
                Component(
                    component_id=f"image_{len(components)}",
                    kind="image",
                    content_ref=ref,
                    label=f"Image {len(components) + 1}",
                )
            )
            content[ref] = image
    for field in TABLE_FIELDS:
        if field not in row or row[field] is None:
            continue
        ref = content_ref(dataset.name, raw_id, field, len(components))
        components.append(
            Component(
                component_id=f"table_{len(components)}",
                kind="table",
                content_ref=ref,
                label=f"Table {len(components) + 1}",
            )
        )
        content[ref] = _describe_value(row[field])
    if len(components) < 2:
        raise RowUnsupported("expected at least 2 mixed-modality components")

    answer = _optional_first_text(row, ANSWER_FIELDS)
    question_ref = content_ref(dataset.name, raw_id, "question")
    content[question_ref] = question
    item = AuditItem(
        item_id=f"{dataset.name}::{raw_id}",
        dataset=dataset.name,
        components=tuple(components),
        question_ref=question_ref,
        gold=answer,
        metadata={"hf_repo": dataset.hf_repo, "split": dataset.split},
    )
    score_kind = get_facet(facet).score_kind
    gold = "1" if score_kind == "llm_judge_gold_match" and answer else normalize_answer(
        score_kind,
        answer,
    )
    return RuntimeExample(
        facet=facet,
        item=item,
        question=question,
        content=content,
        score_kind=score_kind,
        gold_normalized=gold,
    )


def _iter_hf_dataset_rows(
    dataset: DatasetProfile,
    *,
    streaming: bool,
):
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "Install HuggingFace dataset support with `pip install -e '.[hf]'` "
            "or use `--items-jsonl` with normalized AuditItem records."
        ) from exc

    args: list[Any] = [dataset.hf_repo]
    kwargs: dict[str, Any] = {"streaming": streaming}
    if dataset.revision:
        kwargs["revision"] = dataset.revision
    if dataset.config:
        args.append(dataset.config)
    if dataset.split.endswith((".jsonl", ".json", ".parquet")):
        kwargs["data_files"] = dataset.split
        kwargs["split"] = "train"
    else:
        kwargs["split"] = dataset.split

    rows = load_dataset(*args, **kwargs)
    yield from rows


def _content_from_audit_item(item: AuditItem) -> dict[str, str]:
    content = {}
    if item.question_ref:
        content[item.question_ref] = item.question_ref
    component_ids = [component.component_id for component in item.components]
    for idx, component in enumerate(item.components):
        if component.kind == "choice" and idx < len(item.choices):
            content[component.content_ref] = item.choices[idx]
        else:
            content[component.content_ref] = component.content_ref
    if item.gold in component_ids:
        content["gold_component_idx"] = str(component_ids.index(str(item.gold)))
    return content


def _gold_from_item(item: AuditItem, score_kind: str) -> str | None:
    if score_kind == "option_content_idx":
        component_ids = [component.component_id for component in item.components]
        if item.gold in component_ids:
            return str(component_ids.index(str(item.gold)))
        return _index_text(item.gold, len(item.choices) or len(item.components))
    if score_kind == "mcq_letter":
        return _gold_letter(item.gold, list(item.choices))
    if score_kind == "llm_judge_gold_match" and item.gold:
        return "1"
    return normalize_answer(score_kind, item.gold)


def _completed_keys(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str, int]]:
    return {
        (
            str(record.get("model") or ""),
            str(record.get("facet") or ""),
            str(record["item_id"]),
            int(record.get("ordering_idx", 0)),
        )
        for record in records
        if "item_id" in record
    }


def _trial_key(model_name: str, manifest_row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        model_name,
        str(manifest_row["facet"]),
        str(manifest_row["item_id"]),
        int(manifest_row["ordering_idx"]),
    )


def _first_present(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, Any]:
    for field in fields:
        if field in row and row[field] is not None:
            return field, row[field]
    raise RowUnsupported(f"missing any of fields {list(fields)}")


def _optional_first(row: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if field in row and row[field] is not None:
            return row[field]
    return None


def _first_text(row: Mapping[str, Any], fields: Sequence[str]) -> str:
    value = _optional_first(row, fields)
    if value is None:
        raise RowUnsupported(f"missing any of fields {list(fields)}")
    text = _describe_value(value).strip()
    if not text:
        raise RowUnsupported(f"empty text for fields {list(fields)}")
    return text


def _optional_first_text(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    value = _optional_first(row, fields)
    if value is None:
        return None
    text = _describe_value(value).strip()
    return text or None


def _row_id(row: Mapping[str, Any], row_idx: int) -> str:
    for field in ("id", "question_id", "questionId", "uid", "qid", "problem_id", "sample_id"):
        if field in row and row[field] is not None:
            return str(row[field])
    return str(row_idx)


def _audited_n_int(value: int | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value)
    digits = []
    current = []
    for char in text:
        if char.isdigit():
            current.append(char)
        elif current:
            digits.append(int("".join(current)))
            current = []
    if current:
        digits.append(int("".join(current)))
    return digits[-1] if digits else None


def _maybe_choices(row: Mapping[str, Any]) -> list[str]:
    value = _optional_first(row, CHOICE_FIELDS)
    if value is None:
        return []
    return _text_sequence(value)


def _text_sequence(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        for key in ("text", "label", "choices", "options", "items"):
            if key in value and key != "label":
                return _text_sequence(value[key])
        if "title" in value and "sentences" in value:
            titles = _as_sequence(value["title"])
            sentences = _as_sequence(value["sentences"])
            return [
                _join_nonempty([_describe_value(title), _describe_value(sentence)])
                for title, sentence in zip(titles, sentences, strict=False)
            ]
        for key in TEXT_KEYS:
            if key in value:
                return [_describe_value(value[key])]
        return [_describe_value(value)]
    out = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            if "title" in item and any(key in item for key in TEXT_KEYS):
                text = _join_nonempty(
                    [_describe_value(item.get("title")), *_text_values_from_mapping(item)]
                )
            else:
                text_values = _text_values_from_mapping(item)
                text = _join_nonempty(text_values) if text_values else _describe_value(item)
            out.append(text)
        else:
            out.append(_describe_value(item))
    return out


def _text_values_from_mapping(value: Mapping[str, Any]) -> list[str]:
    texts = []
    for key in TEXT_KEYS:
        if key in value and value[key] is not None:
            texts.extend(_describe_value(item) for item in _as_sequence(value[key]))
    return texts


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if "items" in value:
            return list(_as_sequence(value["items"]))
        if "text" in value and isinstance(value["text"], Sequence) and not isinstance(
            value["text"],
            str,
        ):
            return list(value["text"])
        return [value]
    if isinstance(value, str) or not isinstance(value, Sequence):
        return [value]
    return list(value)


def _answer_to_index(answer: Any, choices: Sequence[str]) -> int:
    idx = _index_text(answer, len(choices))
    if idx is not None:
        return int(idx)
    answer_text = _describe_value(answer).strip()
    for idx, choice in enumerate(choices):
        if choice.strip().lower() == answer_text.lower():
            return idx
    for idx, choice in enumerate(choices):
        if choice.strip().lower().startswith(answer_text.lower()[:20]):
            return idx
    raise RowUnsupported(f"cannot map answer {answer!r} to choices")


def _index_text(value: Any, n: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, int) and 0 <= value < n:
        return str(value)
    text = str(value).strip()
    if text.isdigit():
        raw = int(text)
        if 0 <= raw < n:
            return str(raw)
        if 1 <= raw <= n:
            return str(raw - 1)
    if len(text) == 1 and text.isalpha():
        idx = ord(text.upper()) - ord("A")
        if 0 <= idx < n:
            return str(idx)
    return None


def _gold_letter(answer: Any, choices: Sequence[str]) -> str | None:
    if answer is None:
        return None
    if isinstance(answer, int) and 0 <= answer < max(1, len(choices)):
        return chr(ord("A") + answer)
    idx = _index_text(answer, len(choices) or 26)
    if idx is not None:
        return chr(ord("A") + int(idx))
    text = str(answer).strip().upper()
    if len(text) == 1 and "A" <= text <= "Z":
        return text
    for idx, choice in enumerate(choices):
        if choice.strip().lower() == str(answer).strip().lower():
            return chr(ord("A") + idx)
    return None


def _display_letter_for_source_index(
    source_index: str | None,
    permutation: Sequence[Any],
) -> str | None:
    if source_index is None:
        return None
    try:
        source = int(source_index)
        slot = [int(item) for item in permutation].index(source)
    except (ValueError, TypeError):
        return None
    return chr(ord("A") + slot)


def _extract_pipeline_text(outputs: Any) -> str:
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, Mapping):
            for key in ("generated_text", "text"):
                if key in first:
                    return str(first[key])
        return str(first)
    return str(outputs)


def _pieces_to_hf_content(pieces: Sequence[PromptPiece]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for piece in pieces:
        if isinstance(piece, PromptImagePiece):
            content.append({"type": "image", "image": piece.image})
        else:
            content.append({"type": "text", "text": piece.render()})
    return content


def _pieces_to_google_parts(pieces: Sequence[PromptPiece]) -> list[Any]:
    from google.genai import types  # type: ignore

    parts = []
    for piece in pieces:
        if isinstance(piece, PromptImagePiece):
            parts.append(
                types.Part.from_bytes(data=_pil_to_png_bytes(piece.image), mime_type="image/png")
            )
        else:
            parts.append(types.Part.from_text(text=piece.render()))
    return parts


def _pieces_to_openai_responses_content(pieces: Sequence[PromptPiece]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for piece in pieces:
        if isinstance(piece, PromptImagePiece):
            blocks.append({"type": "input_image", "image_url": _pil_to_data_url(piece.image)})
        else:
            blocks.append({"type": "input_text", "text": piece.render()})
    return blocks


def _pieces_to_openai_chat_content(pieces: Sequence[PromptPiece]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for piece in pieces:
        if isinstance(piece, PromptImagePiece):
            blocks.append(
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(piece.image)}}
            )
        else:
            blocks.append({"type": "text", "text": piece.render()})
    return blocks


def _pieces_to_anthropic_content(pieces: Sequence[PromptPiece]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for piece in pieces:
        if isinstance(piece, PromptImagePiece):
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(_pil_to_png_bytes(piece.image)).decode(
                            "ascii"
                        ),
                    },
                }
            )
        else:
            blocks.append({"type": "text", "text": piece.render()})
    return blocks


def _pil_to_data_url(image: Any) -> str:
    return (
        "data:image/png;base64,"
        + base64.standard_b64encode(_pil_to_png_bytes(image)).decode("ascii")
    )


def _pil_to_png_bytes(image: Any) -> bytes:
    image = _coerce_pil_image(image)
    if image is None:
        raise ValueError("expected a PIL-compatible image")
    buf = io.BytesIO()
    img = image.convert("RGB") if getattr(image, "mode", None) not in {"RGB", "RGBA"} else image
    img.save(buf, format="PNG")
    return buf.getvalue()


def _coerce_pil_image(value: Any) -> Any | None:
    if value is None:
        return None
    if hasattr(value, "save") and hasattr(value, "convert"):
        return value
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        return None
    if isinstance(value, Mapping):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"]))
        if value.get("path"):
            return Image.open(value["path"])
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value))
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if path.exists():
            return Image.open(path)
    return None


def _split_thinking(raw: str) -> tuple[str, str | None]:
    start = raw.find("<think>")
    end = raw.find("</think>")
    if start == -1 and end == -1:
        return raw.strip(), None
    if start == -1 and end != -1:
        return raw[end + len("</think>") :].strip(), raw[:end].strip() or None
    if end == -1:
        return raw[:start].strip(), raw[start + len("<think>") :].strip() or None
    thinking = raw[start + len("<think>") : end].strip()
    answer = (raw[:start] + raw[end + len("</think>") :]).strip()
    return answer, thinking or None


def _first_env(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _describe_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, Mapping):
        if "path" in value and value["path"]:
            return str(value["path"])
        if "bytes" in value and value["bytes"]:
            return f"<bytes:{len(value['bytes'])}>"
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple)):
        return "\n".join(_describe_value(item) for item in value)
    filename = getattr(value, "filename", None)
    if filename:
        return str(filename)
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _join_nonempty(values: Sequence[str | None]) -> str:
    return "\n".join(str(value).strip() for value in values if value and str(value).strip())


def _prompt_hash(bundle: PromptBundle) -> str:
    h = hashlib.sha256()
    h.update(bundle.text.encode("utf-8"))
    for piece in bundle.pieces:
        h.update(b"|")
        if isinstance(piece, PromptImagePiece):
            h.update(b"<image>")
            h.update(_describe_value(piece.image).encode("utf-8"))
        else:
            h.update(piece.render().encode("utf-8"))
    return h.hexdigest()[:16]


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
