"""Validation helpers for normalized Facet-Probe items."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, get_args

from facet_probe.facets import get_facet, sample_permutations
from facet_probe.schema import AuditItem, ComponentKind

VALID_COMPONENT_KINDS = set(get_args(ComponentKind))


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    item_id: str
    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    n_items: int
    n_components: int
    issues: tuple[ValidationIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        obj = asdict(self)
        obj["n_errors"] = sum(1 for issue in self.issues if issue.severity == "error")
        obj["n_warnings"] = sum(1 for issue in self.issues if issue.severity == "warning")
        return obj


def validate_audit_items(
    items: list[AuditItem | dict[str, Any]],
    *,
    facet: str | None = None,
    k: int = 6,
) -> ValidationReport:
    parsed = [
        item if isinstance(item, AuditItem) else AuditItem.from_mapping(item)
        for item in items
    ]
    issues: list[ValidationIssue] = []
    seen_item_ids: set[str] = set()
    facet_spec = get_facet(facet) if facet else None

    for item in parsed:
        if not item.item_id:
            issues.append(_issue("error", "<missing>", "missing_item_id", "item_id is required"))
        if item.item_id in seen_item_ids:
            issues.append(
                _issue("error", item.item_id, "duplicate_item_id", "item_id is duplicated")
            )
        seen_item_ids.add(item.item_id)
        if not item.dataset:
            issues.append(_issue("warning", item.item_id, "missing_dataset", "dataset is empty"))
        if not item.components:
            issues.append(
                _issue(
                    "error",
                    item.item_id,
                    "empty_components",
                    "at least one orderable component is required",
                )
            )
            continue

        component_ids: set[str] = set()
        for component in item.components:
            if component.component_id in component_ids:
                issues.append(
                    _issue(
                        "error",
                        item.item_id,
                        "duplicate_component_id",
                        f"component_id {component.component_id!r} is duplicated",
                    )
                )
            component_ids.add(component.component_id)
            if component.kind not in VALID_COMPONENT_KINDS:
                issues.append(
                    _issue(
                        "error",
                        item.item_id,
                        "unknown_component_kind",
                        f"component kind {component.kind!r} is not registered",
                    )
                )
            if not component.content_ref:
                issues.append(
                    _issue(
                        "warning",
                        item.item_id,
                        "missing_content_ref",
                        f"component {component.component_id!r} has no content_ref",
                    )
                )

        if facet_spec is not None:
            issues.extend(_facet_issues(item, facet=facet, k=k))

    ok = not any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        ok=ok,
        n_items=len(parsed),
        n_components=sum(len(item.components) for item in parsed),
        issues=tuple(issues),
    )


def _facet_issues(item: AuditItem, *, facet: str | None, k: int) -> list[ValidationIssue]:
    if facet is None:
        return []
    issues: list[ValidationIssue] = []
    n_components = len(item.components)
    if n_components < 2:
        issues.append(
            _issue(
                "error",
                item.item_id,
                "facet_needs_multiple_components",
                f"{facet} requires at least two orderable components",
            )
        )
    if facet == "option_order":
        if len(item.choices) < 2:
            issues.append(
                _issue(
                    "error",
                    item.item_id,
                    "option_order_missing_choices",
                    "option_order requires at least two choices",
                )
            )
        if item.gold is None:
            issues.append(
                _issue(
                    "warning",
                    item.item_id,
                    "missing_gold",
                    "gold label is recommended for accuracy reporting",
                )
            )
    elif facet == "image_set_order" and not any(c.kind == "image" for c in item.components):
        issues.append(
            _issue(
                "warning",
                item.item_id,
                "image_facet_without_image_kind",
                "image_set_order usually expects image components",
            )
        )
    elif facet == "mixed_modality_order":
        kinds = {component.kind for component in item.components}
        if len(kinds) < 2:
            issues.append(
                _issue(
                    "warning",
                    item.item_id,
                    "single_modality_components",
                    "mixed_modality_order usually expects at least two component kinds",
                )
            )

    try:
        sample_permutations(n_components, k=k, item_id=item.item_id)
    except ValueError as exc:
        issues.append(_issue("error", item.item_id, "permutation_error", str(exc)))
    return issues


def _issue(severity: str, item_id: str, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(severity=severity, item_id=item_id, code=code, message=message)
