"""Adapter templates for common Facet-Probe dataset and model workflows.

These helpers are intentionally small and conservative. They create normalized
``AuditItem`` records with stable content references, while upstream dataset
content stays in the caller's runtime environment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

from facet_probe.prompts import build_mcq_question_block, render_text_component
from facet_probe.schema import AuditItem, Component, ComponentKind

ContentResolver = Callable[[Component], str]


def content_ref(dataset: str, item_id: str, field: str, index: int | None = None) -> str:
    """Build an opaque stable reference to upstream content."""

    parts = [quote(str(dataset), safe=""), quote(str(item_id), safe=""), quote(field, safe="")]
    if index is not None:
        parts.append(str(index))
    return "facet-probe://" + "/".join(parts)


def mcq_audit_item(
    row: Mapping[str, Any],
    *,
    dataset: str,
    item_id_field: str = "id",
    question_field: str = "question",
    choices_field: str = "choices",
    gold_field: str | None = "answer",
    choice_text_field: str = "text",
) -> AuditItem:
    """Template for multiple-choice datasets such as MMLU-Pro or CSQA."""

    item_id = str(row[item_id_field])
    choices = _text_sequence(row[choices_field], text_field=choice_text_field)
    components = tuple(
        Component(
            component_id=f"choice_{idx}",
            kind="choice",
            content_ref=content_ref(dataset, item_id, choices_field, idx),
            label=chr(ord("A") + idx),
        )
        for idx in range(len(choices))
    )
    return AuditItem(
        item_id=f"{dataset}::{item_id}",
        dataset=dataset,
        components=components,
        question_ref=content_ref(dataset, item_id, question_field),
        choices=tuple(choices),
        gold=_optional_text(row, gold_field),
    )


def evidence_list_audit_item(
    row: Mapping[str, Any],
    *,
    dataset: str,
    item_id_field: str = "id",
    evidence_field: str = "evidence",
    question_field: str | None = "question",
    gold_field: str | None = "answer",
    text_field: str = "text",
) -> AuditItem:
    """Template for evidence-chunk datasets with a flat evidence list."""

    item_id = str(row[item_id_field])
    evidence = _text_sequence(row[evidence_field], text_field=text_field)
    components = tuple(
        Component(
            component_id=f"evidence_{idx}",
            kind="text",
            content_ref=content_ref(dataset, item_id, evidence_field, idx),
            label=f"Evidence {idx + 1}",
        )
        for idx in range(len(evidence))
    )
    return AuditItem(
        item_id=f"{dataset}::{item_id}",
        dataset=dataset,
        components=components,
        question_ref=_optional_ref(dataset, item_id, question_field),
        gold=_optional_text(row, gold_field),
    )


def image_list_audit_item(
    row: Mapping[str, Any],
    *,
    dataset: str,
    item_id_field: str = "id",
    images_field: str = "images",
    question_field: str | None = "question",
    choices_field: str | None = "choices",
    gold_field: str | None = "answer",
    choice_text_field: str = "text",
) -> AuditItem:
    """Template for multi-image VQA datasets."""

    item_id = str(row[item_id_field])
    images = _as_sequence(row[images_field])
    choices = (
        tuple(_text_sequence(row[choices_field], text_field=choice_text_field))
        if choices_field and choices_field in row
        else ()
    )
    components = tuple(
        Component(
            component_id=f"image_{idx}",
            kind="image",
            content_ref=content_ref(dataset, item_id, images_field, idx),
            label=f"Image {idx + 1}",
        )
        for idx in range(len(images))
    )
    return AuditItem(
        item_id=f"{dataset}::{item_id}",
        dataset=dataset,
        components=components,
        question_ref=_optional_ref(dataset, item_id, question_field),
        choices=choices,
        gold=_optional_text(row, gold_field),
    )


def mixed_modality_audit_item(
    row: Mapping[str, Any],
    *,
    dataset: str,
    component_fields: Sequence[tuple[str, ComponentKind]],
    item_id_field: str = "id",
    question_field: str | None = "question",
    gold_field: str | None = "answer",
) -> AuditItem:
    """Template for rows with text/image/table component sequences."""

    item_id = str(row[item_id_field])
    components: list[Component] = []
    for field, kind in component_fields:
        if field not in row:
            continue
        for idx, _value in enumerate(_as_sequence(row[field])):
            components.append(
                Component(
                    component_id=f"{field}_{idx}",
                    kind=kind,
                    content_ref=content_ref(dataset, item_id, field, idx),
                    label=f"{field} {idx + 1}",
                )
            )
    return AuditItem(
        item_id=f"{dataset}::{item_id}",
        dataset=dataset,
        components=tuple(components),
        question_ref=_optional_ref(dataset, item_id, question_field),
        gold=_optional_text(row, gold_field),
    )


def render_ordered_text_prompt(
    item: AuditItem,
    ordered_component_ids: Sequence[str],
    *,
    resolve_content: ContentResolver,
    question: str | None = None,
    instruction: str = "Answer using only the supplied evidence.",
) -> str:
    """Render an ordered text prompt for a provider/local-model adapter."""

    by_id = {component.component_id: component for component in item.components}
    component_index = {
        component.component_id: idx
        for idx, component in enumerate(item.components)
    }
    missing = [component_id for component_id in ordered_component_ids if component_id not in by_id]
    if missing:
        raise KeyError(f"unknown component ids for {item.item_id}: {missing}")
    ordered_components = [by_id[component_id] for component_id in ordered_component_ids]

    if question is not None and item.choices and all(
        component.kind == "choice" for component in ordered_components
    ):
        ordered_choices = [
            item.choices[component_index[component.component_id]]
            for component in ordered_components
        ]
        return "\n".join([build_mcq_question_block(question, ordered_choices), instruction])

    lines = []
    if question is not None:
        if item.choices:
            lines.append(build_mcq_question_block(question, item.choices))
        else:
            lines.append("Question: " + question.strip())
    elif item.question_ref:
        lines.append("Question ref: " + item.question_ref)
    lines.append("Evidence:")
    for idx, component_id in enumerate(ordered_component_ids, start=1):
        component = by_id[component_id]
        text = resolve_content(component)
        label = component.label or f"{component.kind} {idx}"
        lines.append(f"[{idx}] " + render_text_component(label, text))
    lines.append(instruction)
    return "\n".join(lines)


def _optional_text(row: Mapping[str, Any], field: str | None) -> str | None:
    if field is None or field not in row or row[field] is None:
        return None
    return str(row[field])


def _optional_ref(dataset: str, item_id: str, field: str | None) -> str | None:
    if field is None:
        return None
    return content_ref(dataset, item_id, field)


def _as_sequence(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        if "items" in value:
            return list(value["items"])
        if "text" in value:
            return list(value["text"])
        return list(value.values())
    if isinstance(value, str) or not isinstance(value, Sequence):
        return [value]
    return list(value)


def _text_sequence(value: Any, *, text_field: str) -> list[str]:
    if isinstance(value, Mapping) and text_field in value:
        return [str(item) for item in _as_sequence(value[text_field])]
    out = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping) and text_field in item:
            out.append(str(item[text_field]))
        else:
            out.append(str(item))
    return out
