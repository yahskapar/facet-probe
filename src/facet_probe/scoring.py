"""Answer parsing and scoring helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence

from facet_probe.facets import source_option_index

_ANSWER_PREFIX_RE = re.compile(r"(?im)^\s*(?:final\s+answer|answer)\s*[:\-]\s*(.+?)\s*$")
_NUMBER_RE = re.compile(r"-?\d+(?:[,.]\d+)*")


def parse_answer_letter(raw: str | None, n_choices: int) -> str | None:
    """Parse a single MCQ letter from a model response."""

    if not raw:
        return None
    text = str(raw).strip()
    valid = {chr(ord("A") + i) for i in range(max(0, n_choices))}

    for match in _ANSWER_PREFIX_RE.finditer(text):
        candidate = match.group(1).strip().upper()
        if candidate[:1] in valid:
            return candidate[:1]

    # Common bare-letter final line.
    lines = [line.strip().upper() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        if len(last) == 1 and last in valid:
            return last
        m = re.match(r"^\(?([A-Z])\)?(?:[.)\s]|$)", last)
        if m and m.group(1) in valid:
            return m.group(1)

    m = re.search(r"\b([A-Z])\b", text.upper())
    if m and m.group(1) in valid:
        return m.group(1)
    return None


def normalize_text(raw: str | None) -> str | None:
    """Normalize short free-form answers for exact-match style scoring."""

    if raw is None:
        return None
    text = str(raw).strip()
    match = None
    for candidate in _ANSWER_PREFIX_RE.finditer(text):
        match = candidate
    if match is not None:
        text = match.group(1)
    elif "\n" in text:
        nonempty = [line.strip() for line in text.splitlines() if line.strip()]
        text = nonempty[-1] if nonempty else ""
    text = text.lower().strip()
    text = re.sub(r"^(?:the\s+answer\s+is|the\s+final\s+answer\s+is)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .'\"`")
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    return text or None


def last_number(raw: str | None) -> str | None:
    """Return the last numeric string in a response, normalized for comparison."""

    if not raw:
        return None
    nums = _NUMBER_RE.findall(str(raw))
    if not nums:
        return None
    number = nums[-1].replace(",", "")
    if "." in number:
        number = number.rstrip("0").rstrip(".") or "0"
    return number


def normalize_answer(
    score_kind: str,
    raw: str | None,
    *,
    n_choices: int = 0,
    permutation: Sequence[int] = (),
) -> str | None:
    """Normalize a model answer for one of the Facet-Probe scoring modes."""

    if score_kind == "mcq_letter":
        return parse_answer_letter(raw, n_choices)
    if score_kind == "option_content_idx":
        letter = parse_answer_letter(raw, n_choices)
        return source_option_index(letter, tuple(permutation))
    if score_kind == "exact_match":
        return normalize_text(raw)
    if score_kind == "exact_number":
        return last_number(raw)
    if score_kind == "function_name":
        if not raw:
            return None
        match = re.search(r'["\']name["\']\s*:\s*["\']([A-Za-z_][\w.]*)["\']', str(raw))
        if match:
            return match.group(1)
        match = re.search(r"\b([A-Za-z_][\w.]*)\s*\(", str(raw))
        return match.group(1) if match else None
    if score_kind == "llm_judge_gold_match":
        if raw is None:
            return None
        text = str(raw).strip().lower()
        if text in {"true", "correct", "1", "yes", "equivalent", "gold_match"}:
            return "1"
        if text in {"false", "incorrect", "0", "no", "not_equivalent", "not_gold_match"}:
            return "0"
        return text or None
    raise ValueError(f"unknown score_kind: {score_kind!r}")


def score_answer(answer_normalized: str | None, gold_normalized: str | None) -> bool | None:
    """Return correctness for normalized answers, or None when either side is missing."""

    if answer_normalized is None or gold_normalized is None:
        return None
    return str(answer_normalized) == str(gold_normalized)
