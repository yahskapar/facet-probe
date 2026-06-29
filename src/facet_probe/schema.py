"""Canonical records used by Facet-Probe.

The public schema intentionally separates dataset content from ordering metadata.
Release artifacts should store stable item IDs, permutations, normalized answers,
and scores; upstream dataset text/images are loaded from their original sources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ComponentKind = Literal["text", "image", "table", "mixed", "choice", "document"]


@dataclass(frozen=True)
class Component:
    """One orderable component in a benchmark item."""

    component_id: str
    kind: ComponentKind
    content_ref: str
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> Component:
        return cls(
            component_id=str(row["component_id"]),
            kind=row.get("kind", "text"),
            content_ref=str(row.get("content_ref") or row["component_id"]),
            label=row.get("label"),
            metadata={k: v for k, v in row.items() if k not in _COMPONENT_FIELDS},
        )


@dataclass(frozen=True)
class AuditItem:
    """Dataset-normalized item with all orderable units exposed."""

    item_id: str
    dataset: str
    components: tuple[Component, ...]
    question_ref: str | None = None
    choices: tuple[str, ...] = ()
    gold: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> AuditItem:
        components = tuple(
            component if isinstance(component, Component) else Component.from_mapping(component)
            for component in row.get("components", [])
        )
        return cls(
            item_id=str(row["item_id"]),
            dataset=str(row.get("dataset") or ""),
            components=components,
            question_ref=row.get("question_ref"),
            choices=tuple(str(choice) for choice in row.get("choices", ())),
            gold=None if row.get("gold") is None else str(row.get("gold")),
            metadata={k: v for k, v in row.items() if k not in _AUDIT_ITEM_FIELDS},
        )


@dataclass(frozen=True)
class TrialRecord:
    """One model call under one ordering."""

    facet: str
    dataset: str
    item_id: str
    ordering_idx: int
    permutation: tuple[int, ...]
    model: str | None = None
    answer_normalized: str | None = None
    gold_normalized: str | None = None
    correct: bool | None = None
    score_kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> TrialRecord:
        """Build a TrialRecord from a JSONL-style mapping."""

        answer = row.get("answer_normalized")
        if answer is None:
            answer = row.get("answer_letter")
        if answer is None:
            answer = row.get("answer")

        perm = row.get("permutation") or ()
        return cls(
            facet=str(row.get("facet") or ""),
            dataset=str(row.get("dataset") or ""),
            item_id=str(row["item_id"]),
            ordering_idx=int(row.get("ordering_idx", 0)),
            permutation=tuple(int(x) for x in perm),
            model=row.get("model"),
            answer_normalized=None if answer is None else str(answer),
            gold_normalized=(
                None if row.get("gold_normalized") is None else str(row.get("gold_normalized"))
            ),
            correct=_coerce_bool(row.get("correct")),
            score_kind=row.get("score_kind"),
            metadata={k: v for k, v in row.items() if k not in _TRIAL_FIELDS},
        )


def _coerce_bool(value: Any) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "correct"}:
        return True
    if text in {"false", "0", "no", "n", "incorrect"}:
        return False
    return None


_TRIAL_FIELDS = {
    "facet",
    "dataset",
    "item_id",
    "ordering_idx",
    "permutation",
    "model",
    "answer",
    "answer_letter",
    "answer_normalized",
    "gold_normalized",
    "correct",
    "score_kind",
}

_COMPONENT_FIELDS = {
    "component_id",
    "kind",
    "content_ref",
    "label",
}

_AUDIT_ITEM_FIELDS = {
    "item_id",
    "dataset",
    "components",
    "question_ref",
    "choices",
    "gold",
}
