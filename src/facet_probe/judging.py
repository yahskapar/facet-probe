"""Semantic-equivalence judging for mixed-modality free-form outputs."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from facet_probe.metrics import write_csv, write_json
from facet_probe.profiles import ModelProfile
from facet_probe.runner import PromptTextPiece, create_model_adapter
from facet_probe.scoring import normalize_answer

JUDGE_SYSTEM = (
    "You are a careful evaluator comparing multiple answers to the same "
    "multimodal question. Multiple answers were produced by the same model "
    "across different orderings of the same evidence. Your job: judge whether "
    "the answers are semantically equivalent, meaning same content possibly "
    "paraphrased, or whether they reflect a content flip with different facts, "
    "different conclusions, or different referents."
)

JUDGE_PROMPT_TEMPLATE = """A multimodal model was asked the same question {k} times.
Each time, the underlying evidence was the same but was presented in a different order.
The model produced these {k} answers:

QUESTION: {question}

{answers_block}

Decide ONE label that best characterizes the set of answers:

A. EQUIVALENT - all {k} answers say the same thing. Paraphrased wording is fine
if the facts, conclusion, and referents are the same.
B. PARTIAL - some answers agree with each other but at least one differs in content,
not just wording.
C. FLIP - the answers reflect substantially different content across orderings,
such as different facts, conclusions, or referents.
D. UNPARSABLE - at least one answer is too short, malformed, or off-task to judge equivalence.

Respond strictly in this format:
Verdict: <A|B|C|D>
Reason: <one short sentence, <=30 words>
"""
ProgressCallback = Callable[[str], None]


def judge_mixed_trials(
    records: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    judge_model: ModelProfile | None = None,
    mock_judge: bool = False,
    max_new_tokens: int | None = None,
    limit_items: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Judge mixed-modality free-form outputs and write semantic-flip artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "judgments": output / "mixed_semantic_judgments.jsonl",
        "summary_json": output / "mixed_semantic_summary.json",
        "summary_csv": output / "mixed_semantic_summary.csv",
    }

    _progress(progress_callback, "grouping mixed_modality_order trial records")
    groups = _mixed_groups(records)
    if limit_items is not None:
        groups = dict(list(groups.items())[:limit_items])
    if not groups:
        raise ValueError("no mixed_modality_order records found to judge")
    if not mock_judge and judge_model is None:
        raise ValueError("provide a judge_model or pass mock_judge=True")

    effective_max_tokens = max_new_tokens
    if effective_max_tokens is None:
        generation = judge_model.generation if judge_model is not None else {}
        effective_max_tokens = int(generation.get("max_output_tokens", 512))

    _progress(
        progress_callback,
        (
            f"judging {len(groups)} mixed item group(s) "
            f"with {'mock' if mock_judge else judge_model.name}"
        ),
    )
    adapter = None if mock_judge else create_model_adapter(judge_model)  # type: ignore[arg-type]
    judgments = []
    progress_every = _progress_interval(len(groups))
    try:
        for idx, (key, rows) in enumerate(groups.items(), start=1):
            if _should_emit_progress(idx - 1, len(groups), progress_every):
                _progress(
                    progress_callback,
                    _judge_progress_message("running", idx, len(groups), key),
                )
            started = time.monotonic()
            judgments.append(
                _judge_one_group(
                    key,
                    rows,
                    adapter=adapter,
                    judge_model=judge_model,
                    mock_judge=mock_judge,
                    max_new_tokens=effective_max_tokens,
                )
            )
            if _should_emit_progress(idx, len(groups), progress_every):
                _progress(
                    progress_callback,
                    _judge_progress_message(
                        "completed",
                        idx,
                        len(groups),
                        key,
                        elapsed_seconds=time.monotonic() - started,
                    ),
                )
    finally:
        if adapter is not None:
            adapter.close()
            _progress(progress_callback, "closed judge adapter")

    _progress(progress_callback, "writing mixed-modality judge summary files")
    _write_jsonl(files["judgments"], judgments)
    summary_rows = _semantic_summary_rows(groups, judgments)
    write_csv(files["summary_csv"], summary_rows)
    status = {
        "status": "completed",
        "n_mixed_items": len(groups),
        "n_judgments": len(judgments),
        "judge_model": "mock" if mock_judge else (judge_model.name if judge_model else None),
        "summary": summary_rows,
        "files": {name: str(path) for name, path in files.items()},
    }
    write_json(files["summary_json"], status)
    return status


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _progress_interval(total: int) -> int:
    if total <= 20:
        return 1
    return max(1, total // 100)


def _should_emit_progress(done: int, total: int, interval: int) -> bool:
    if total <= 0:
        return False
    return done == 0 or done == total or done % interval == 0


def _judge_progress_message(
    action: str,
    done: int,
    total: int,
    key: tuple[str, str, str],
    *,
    elapsed_seconds: float | None = None,
) -> str:
    dataset, model, item_id = key
    pct = 100.0 * done / total if total else 100.0
    parts = [
        f"{action} judgment {done}/{total} ({pct:.1f}%)",
        f"dataset={dataset}",
        f"model={model}",
        f"item={item_id}",
    ]
    if elapsed_seconds is not None:
        parts.append(f"elapsed={elapsed_seconds:.1f}s")
    return " ".join(parts)


def _mixed_groups(
    records: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if str(record.get("facet") or "") != "mixed_modality_order":
            continue
        key = (
            str(record.get("dataset") or ""),
            str(record.get("model") or ""),
            str(record.get("item_id") or ""),
        )
        grouped[key].append(dict(record))
    return {
        key: sorted(rows, key=lambda row: int(row.get("ordering_idx", 0)))
        for key, rows in sorted(grouped.items())
    }


def _judge_one_group(
    key: tuple[str, str, str],
    rows: Sequence[Mapping[str, Any]],
    *,
    adapter: Any,
    judge_model: ModelProfile | None,
    mock_judge: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    dataset, model, item_id = key
    answers = [_answer_text(row) for row in rows]
    question = _question_text(rows)
    t0 = time.time()
    if mock_judge:
        verdict, reason, raw = _mock_verdict(answers)
    else:
        raw = _call_judge_adapter(
            adapter,
            question=question,
            answers=answers,
            max_new_tokens=max_new_tokens,
        )
        verdict, reason = parse_judge_response(raw)
    latency = round(time.time() - t0, 3)
    return {
        "schema_version": 1,
        "facet": "mixed_modality_order",
        "dataset": dataset,
        "model": model,
        "item_id": item_id,
        "judge_model": "mock" if mock_judge else (judge_model.name if judge_model else None),
        "verdict": verdict,
        "reason": reason,
        "n_answers_judged": len(answers),
        "text_flip": len({_content_key(answer) for answer in answers if answer}) > 1,
        "latency_sec": latency,
        "judge_raw_output": raw,
    }


def _call_judge_adapter(
    adapter: Any,
    *,
    question: str,
    answers: Sequence[str],
    max_new_tokens: int,
) -> str:
    prompt = _build_judge_prompt(question, answers)
    response = adapter.generate(
        prompt,
        pieces=(PromptTextPiece(prompt),),
        example=_judge_runtime_example(),
        manifest_row={"ordered_component_ids": ["answers"], "permutation": [0]},
        max_new_tokens=max_new_tokens,
    )
    return response.text


def _judge_runtime_example():
    from facet_probe.runner import RuntimeExample
    from facet_probe.schema import AuditItem, Component

    item = AuditItem(
        item_id="judge::mixed_semantic_equivalence",
        dataset="judge",
        components=(Component("answers", "text", "judge://answers", "Answers"),),
        question_ref="judge://question",
    )
    return RuntimeExample(
        facet="mixed_modality_order",
        item=item,
        question=None,
        content={},
        score_kind="exact_match",
        gold_normalized=None,
        system_instruction=JUDGE_SYSTEM,
        answer_instruction="",
    )


def _build_judge_prompt(question: str, answers: Sequence[str]) -> str:
    answers_block = "\n".join(
        f"ANSWER {idx + 1}: {str(answer).strip()}" for idx, answer in enumerate(answers)
    )
    return JUDGE_PROMPT_TEMPLATE.format(
        k=len(answers),
        question=question,
        answers_block=answers_block,
    )


def parse_judge_response(text: str | None) -> tuple[str, str]:
    verdict = "?"
    reason = ""
    for line in str(text or "").splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("VERDICT:"):
            rest = stripped.split(":", 1)[1].strip().upper()
            if rest and rest[0] in {"A", "B", "C", "D"}:
                verdict = rest[0]
        elif upper.startswith("REASON:"):
            reason = stripped.split(":", 1)[1].strip()
    return verdict, reason


def _semantic_summary_rows(
    groups: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
    judgments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    text_flip_by_item = {
        key: _text_flip(rows)
        for key, rows in groups.items()
    }
    by_group: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for judgment in judgments:
        by_group[(str(judgment["dataset"]), str(judgment["model"]))].append(judgment)

    rows = []
    for (dataset, model), items in sorted(by_group.items()):
        counts = Counter(str(item.get("verdict") or "?") for item in items)
        n = len(items)
        semantic_flips = counts["B"] + counts["C"]
        group_keys = [key for key in groups if key[0] == dataset and key[1] == model]
        text_flip = (
            sum(1 for key in group_keys if text_flip_by_item.get(key)) / len(group_keys)
            if group_keys
            else 0.0
        )
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "n_items": n,
                "sem_flip": semantic_flips / n if n else 0.0,
                "text_flip_upper": text_flip,
                "verdict_A": counts["A"],
                "verdict_B": counts["B"],
                "verdict_C": counts["C"],
                "verdict_D": counts["D"],
                "verdict_unknown": counts["?"],
            }
        )
    return rows


def _mock_verdict(answers: Sequence[str]) -> tuple[str, str, str]:
    normalized = [_content_key(answer) for answer in answers if answer]
    if len(normalized) < len(answers):
        return "D", "At least one answer was empty.", "Verdict: D\nReason: Empty answer."
    if len(set(normalized)) <= 1:
        return "A", "All normalized answers match.", "Verdict: A\nReason: Answers match."
    return "C", "Normalized answers differ.", "Verdict: C\nReason: Answers differ."


def _text_flip(rows: Sequence[Mapping[str, Any]]) -> bool:
    answers = {_content_key(_answer_text(row)) for row in rows if _answer_text(row)}
    return len(answers) > 1


def _answer_text(record: Mapping[str, Any]) -> str:
    for field in ("raw_output", "answer_text", "answer_normalized"):
        value = record.get(field)
        if value is not None:
            return str(value)
    return ""


def _question_text(rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        value = row.get("question")
        if value:
            return str(value)[:1000]
    return "[question text not stored in trial JSONL]"


def _content_key(text: str) -> str:
    return normalize_answer("exact_match", text) or ""


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
