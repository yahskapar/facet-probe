"""Ordering-sensitivity metrics."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from facet_probe.schema import TrialRecord


@dataclass(frozen=True)
class ItemMetric:
    facet: str
    dataset: str
    model: str
    item_id: str
    n_orderings: int
    n_parseable: int
    n_distinct_answers: int
    flipped: bool
    osi: float
    modal_answer: str | None
    modal_fraction: float
    accuracy: float | None


@dataclass(frozen=True)
class AuditSummary:
    label: str
    n_items: int
    n_trials: int
    flip_rate: float
    mean_osi: float
    macro_accuracy: float | None
    n_parseable_trials: int


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalized_entropy(counts: Iterable[int], n_classes: int | None = None) -> float:
    counts = [int(c) for c in counts if int(c) > 0]
    total = sum(counts)
    if total == 0:
        return 0.0
    if n_classes is None:
        n_classes = max(2, len(counts))
    if n_classes <= 1:
        return 0.0
    denom = math.log(n_classes)
    if denom == 0:
        return 0.0
    entropy = -sum((c / total) * math.log(c / total) for c in counts)
    return entropy / denom


def _as_trial_records(records: Iterable[dict[str, Any] | TrialRecord]) -> list[TrialRecord]:
    out: list[TrialRecord] = []
    for record in records:
        out.append(record if isinstance(record, TrialRecord) else TrialRecord.from_mapping(record))
    return out


def item_metrics(
    records: Iterable[dict[str, Any] | TrialRecord],
    *,
    n_classes_by_item: dict[str, int] | None = None,
) -> list[ItemMetric]:
    """Compute per-item K-ordering metrics."""

    trials = _as_trial_records(records)
    grouped: dict[tuple[str, str, str, str], list[TrialRecord]] = defaultdict(list)
    for row in trials:
        grouped[(row.facet, row.dataset, row.model or "", row.item_id)].append(row)

    metrics: list[ItemMetric] = []
    for (facet, dataset, model, item_id), rows in sorted(grouped.items()):
        answers = [r.answer_normalized for r in rows if r.answer_normalized is not None]
        counts = Counter(answers)
        modal_answer = None
        modal_fraction = 0.0
        if counts:
            modal_answer, modal_count = counts.most_common(1)[0]
            modal_fraction = modal_count / len(answers)
        n_classes = None
        if n_classes_by_item is not None:
            n_classes = n_classes_by_item.get(item_id)
        correct_values = [r.correct for r in rows if r.correct is not None]
        accuracy = (
            sum(1 for value in correct_values if bool(value)) / len(correct_values)
            if correct_values
            else None
        )
        metrics.append(
            ItemMetric(
                facet=facet,
                dataset=dataset,
                model=model,
                item_id=item_id,
                n_orderings=len(rows),
                n_parseable=len(answers),
                n_distinct_answers=len(counts),
                flipped=len(counts) > 1,
                osi=normalized_entropy(counts.values(), n_classes=n_classes),
                modal_answer=modal_answer,
                modal_fraction=modal_fraction,
                accuracy=accuracy,
            )
        )
    return metrics


def summarize_groups(
    records: Iterable[dict[str, Any] | TrialRecord],
    *,
    group_by: Sequence[str] = ("facet", "dataset", "model"),
) -> list[dict[str, Any]]:
    """Aggregate item-level metrics by facet/dataset/model or another key set."""

    per_item = item_metrics(records)
    grouped: dict[tuple[Any, ...], list[ItemMetric]] = defaultdict(list)
    for metric in per_item:
        key = tuple(getattr(metric, name) for name in group_by)
        grouped[key].append(metric)

    rows: list[dict[str, Any]] = []
    for key, metrics in sorted(grouped.items()):
        accs = [m.accuracy for m in metrics if m.accuracy is not None]
        row = {name: value for name, value in zip(group_by, key, strict=True)}
        row.update(
            n_items=len(metrics),
            flip_rate=sum(1 for m in metrics if m.flipped) / len(metrics),
            mean_osi=sum(m.osi for m in metrics) / len(metrics),
            macro_accuracy=(sum(accs) / len(accs) if accs else None),
            n_parseable_trials=sum(m.n_parseable for m in metrics),
            n_trials=sum(m.n_orderings for m in metrics),
        )
        rows.append(row)
    return rows


def audit_records(
    records: Iterable[dict[str, Any] | TrialRecord],
    *,
    label: str = "audit",
) -> AuditSummary:
    per_item = item_metrics(records)
    if not per_item:
        raise ValueError("no records to audit")
    accs = [m.accuracy for m in per_item if m.accuracy is not None]
    return AuditSummary(
        label=label,
        n_items=len(per_item),
        n_trials=sum(m.n_orderings for m in per_item),
        flip_rate=sum(1 for m in per_item if m.flipped) / len(per_item),
        mean_osi=sum(m.osi for m in per_item) / len(per_item),
        macro_accuracy=sum(accs) / len(accs) if accs else None,
        n_parseable_trials=sum(m.n_parseable for m in per_item),
    )


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def item_metrics_as_dicts(records: Iterable[dict[str, Any] | TrialRecord]) -> list[dict[str, Any]]:
    return [asdict(metric) for metric in item_metrics(records)]
