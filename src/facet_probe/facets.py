"""Facet definitions and deterministic permutation grammar."""

from __future__ import annotations

import hashlib
import itertools
import random
from dataclasses import dataclass
from typing import Literal

FacetName = Literal[
    "option_order",
    "evidence_chunk_order",
    "document_rank_order",
    "image_set_order",
    "mixed_modality_order",
    "dialog_turn_order",
    "few_shot_order",
    "tool_description_order",
]


@dataclass(frozen=True)
class FacetSpec:
    name: str
    unit: str
    score_kind: str
    description: str
    main_paper: bool = True


FACETS: dict[str, FacetSpec] = {
    "option_order": FacetSpec(
        name="option_order",
        unit="answer-option display slots",
        score_kind="option_content_idx",
        description=(
            "Permutes which option content is displayed under each letter; scoring maps "
            "the predicted letter back to the source option index."
        ),
    ),
    "evidence_chunk_order": FacetSpec(
        name="evidence_chunk_order",
        unit="flat evidence passages",
        score_kind="exact_match",
        description="Permutes paragraphs or short passages in a flat evidence list.",
    ),
    "document_rank_order": FacetSpec(
        name="document_rank_order",
        unit="ranked retrieved documents",
        score_kind="exact_match",
        description="Permutes complete retrieved documents while preserving their content.",
    ),
    "image_set_order": FacetSpec(
        name="image_set_order",
        unit="multi-image input slots",
        score_kind="mcq_letter",
        description=(
            "Permutes image inputs; clean summaries apply the position-reference screen."
        ),
    ),
    "mixed_modality_order": FacetSpec(
        name="mixed_modality_order",
        unit="heterogeneous text/image component sequence",
        score_kind="llm_judge_gold_match",
        description="Permutes the whole text-and-image component sequence in free-form RAG.",
    ),
    "dialog_turn_order": FacetSpec(
        name="dialog_turn_order",
        unit="prior dialog turns",
        score_kind="exact_match_or_judge",
        description="Demoted facet: permutes prior conversation turns.",
        main_paper=False,
    ),
    "few_shot_order": FacetSpec(
        name="few_shot_order",
        unit="few-shot demonstrations",
        score_kind="task_specific",
        description="Demoted facet: permutes in-context examples.",
        main_paper=False,
    ),
    "tool_description_order": FacetSpec(
        name="tool_description_order",
        unit="tool descriptions",
        score_kind="function_name",
        description="Null/stress facet: permutes tool descriptions in tool-selection prompts.",
        main_paper=False,
    ),
}


def get_facet(name: str) -> FacetSpec:
    try:
        return FACETS[name]
    except KeyError as exc:
        raise KeyError(f"unknown facet {name!r}; known facets: {sorted(FACETS)}") from exc


def stable_int_seed(*parts: object, seed: int = 42) -> int:
    """Return a process-stable 32-bit seed.

    Python's built-in hash() is intentionally randomized between processes, so
    it should not be used for public permutation manifests.
    """

    h = hashlib.sha256()
    h.update(str(seed).encode("utf-8"))
    for part in parts:
        h.update(b"\x1f")
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "big", signed=False)


def sample_permutations(
    n_components: int,
    *,
    k: int = 6,
    seed: int = 42,
    item_id: str = "",
    include_canonical: bool = True,
) -> list[tuple[int, ...]]:
    """Sample the paper's K-ordering grammar for one item.

    If n! >= K, this samples K distinct permutations without replacement,
    including canonical first by default. If n! < K, all unique permutations are
    cycled to length K, matching the paper protocol for small component counts.
    """

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if k <= 0:
        raise ValueError("k must be positive")

    base = tuple(range(n_components))
    rng = random.Random(stable_int_seed(item_id, n_components, k, seed=seed))

    if n_components <= 7:
        all_perms = list(itertools.permutations(base))
        if len(all_perms) >= k:
            pool = [p for p in all_perms if p != base] if include_canonical else all_perms[:]
            rng.shuffle(pool)
            out = [base, *pool[: k - 1]] if include_canonical else pool[:k]
            return out
        return [all_perms[i % len(all_perms)] for i in range(k)]

    out: list[tuple[int, ...]] = [base] if include_canonical else []
    seen = set(out)
    while len(out) < k:
        candidate = list(base)
        rng.shuffle(candidate)
        perm = tuple(candidate)
        if perm not in seen:
            seen.add(perm)
            out.append(perm)
    return out


def source_option_index(
    answer_letter: str | None,
    permutation: list[int] | tuple[int, ...],
) -> str | None:
    """Map a displayed option letter back to the source option index."""

    if answer_letter is None:
        return None
    letter = answer_letter.strip().upper()
    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        return None
    slot = ord(letter) - ord("A")
    if slot < 0 or slot >= len(permutation):
        return None
    return str(permutation[slot])
