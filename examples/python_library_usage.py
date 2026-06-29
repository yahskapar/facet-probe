#!/usr/bin/env python3
"""Use Facet-Probe from Python instead of the CLI."""

from __future__ import annotations

import json

import facet_probe as fp

ITEMS = [
    fp.mcq_audit_item(
        {
            "id": "001",
            "question": "Which option is the target color?",
            "choices": ["red", "blue", "green", "yellow"],
            "answer": "2",
        },
        dataset="toy_mcq",
    )
]


def main() -> None:
    validation = fp.validate_audit_items(ITEMS, facet="option_order", k=3)
    if not validation.ok:
        raise SystemExit(json.dumps(validation.to_dict(), indent=2, sort_keys=True))

    manifest = fp.trial_manifest_rows(ITEMS, facet="option_order", k=3, seed=42)
    raw_answers = ["C", "B", "C"]
    records = []
    for row, raw_answer in zip(manifest, raw_answers, strict=True):
        answer = fp.normalize_answer(
            "option_content_idx",
            raw_answer,
            n_choices=4,
            permutation=row["permutation"],
        )
        records.append(
            {
                "facet": row["facet"],
                "dataset": row["dataset"],
                "model": "example-model",
                "item_id": row["item_id"],
                "ordering_idx": row["ordering_idx"],
                "permutation": row["permutation"],
                "answer_normalized": answer,
                "gold_normalized": manifest[0]["gold"],
                "correct": fp.score_answer(answer, manifest[0]["gold"]),
                "score_kind": "option_content_idx",
            }
        )

    report = fp.build_evaluation_report(records, label="python-library-example")
    print(
        json.dumps(
            {
                "facet_probe_version": fp.__version__,
                "validation": validation.to_dict(),
                "manifest_rows": len(manifest),
                "first_ordering": manifest[0]["ordered_component_ids"],
                "summary": report["summary"],
                "groups": report["groups"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
