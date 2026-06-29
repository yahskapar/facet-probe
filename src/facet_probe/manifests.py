"""Trial-manifest builders for dataset and model adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from facet_probe.facets import sample_permutations
from facet_probe.schema import AuditItem


def audit_item_from_mapping(row: dict[str, Any]) -> AuditItem:
    return AuditItem.from_mapping(row)


def trial_manifest_rows(
    items: Iterable[AuditItem | dict[str, Any]],
    *,
    facet: str,
    k: int = 6,
    seed: int = 42,
    include_ordered_components: bool = False,
) -> list[dict[str, Any]]:
    """Build deterministic trial-manifest rows from normalized audit items."""

    rows: list[dict[str, Any]] = []
    for item_like in items:
        item = item_like if isinstance(item_like, AuditItem) else AuditItem.from_mapping(item_like)
        n_components = len(item.components)
        if n_components <= 0:
            raise ValueError(f"item {item.item_id!r} has no orderable components")
        permutations = sample_permutations(
            n_components,
            k=k,
            seed=seed,
            item_id=item.item_id,
        )
        component_ids = [component.component_id for component in item.components]
        for ordering_idx, permutation in enumerate(permutations):
            row: dict[str, Any] = {
                "facet": facet,
                "dataset": item.dataset,
                "item_id": item.item_id,
                "ordering_idx": ordering_idx,
                "permutation": list(permutation),
                "component_ids": component_ids,
                "ordered_component_ids": [component_ids[idx] for idx in permutation],
                "question_ref": item.question_ref,
                "gold": item.gold,
            }
            if include_ordered_components:
                ordered = [item.components[idx] for idx in permutation]
                row["ordered_components"] = [asdict(component) for component in ordered]
            rows.append(row)
    return rows
