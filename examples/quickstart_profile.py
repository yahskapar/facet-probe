#!/usr/bin/env python3
"""Profile-driven Facet-Probe quickstart without network or model calls."""

from __future__ import annotations

import json

import facet_probe as fp


def main() -> None:
    paper = fp.paper_profile(
        config_dir="configs",
        models=["gemini-3.1-pro-preview", "qwen3-5-4b"],
    )
    gemini, qwen = paper.models
    judge = fp.judge_profile(config_dir="configs")
    profile = paper

    inspection = fp.build_hf_inspection(
        dataset_id="allenai/ai2_arc",
        config="ARC-Challenge",
        split="validation",
        license="CC-BY-SA-4.0",
        features={"question": "string", "choices": {"text": ["string"]}, "answer": "string"},
    )
    new_dataset = fp.hf_dataset(
        inspection.dataset_id,
        name="arc_challenge",
        config=inspection.config,
        split=inspection.split or "validation",
        facets=inspection.candidate_facets or ("option_order",),
        license=inspection.license,
        filters={"min_choices": 4},
    )
    custom_only = profile.only_datasets(new_dataset, name="arc-challenge-only")
    paper_plus_custom = profile.add_datasets(new_dataset, name="paper-plus-arc-challenge")

    row = {
        "id": "demo-001",
        "question": "Which option is the target color?",
        "choices": ["red", "blue", "green", "yellow"],
        "answer": "2",
    }
    item = fp.mcq_audit_item(row, dataset=new_dataset.name)
    validation = fp.validate_audit_items([item], facet="option_order", k=custom_only.k_orderings)
    if not validation.ok:
        raise SystemExit(json.dumps(validation.to_dict(), indent=2, sort_keys=True))

    manifest = fp.trial_manifest_rows(
        [item],
        facet="option_order",
        k=custom_only.k_orderings,
        seed=custom_only.seed,
        include_ordered_components=True,
    )
    first_prompt = fp.render_ordered_text_prompt(
        item,
        manifest[0]["ordered_component_ids"],
        question=row["question"],
        resolve_content=lambda component: component.content_ref,
    )

    print(
        json.dumps(
            {
                "facet_probe_version": fp.__version__,
                "paper_profile": {
                    "name": paper.name,
                    "models": [model.name for model in paper.models],
                    "datasets": len(paper.datasets),
                    "k_orderings": paper.k_orderings,
                    "seed": paper.seed,
                },
                "closed_source_model": gemini.to_dict(),
                "open_weight_model": qwen.to_dict(),
                "mixed_semantic_judge": judge.to_dict(),
                "judge_env_status": judge.env_status(),
                "provider_env_status": profile.provider_status(),
                "candidate_facets_for_new_dataset": inspection.candidate_facets,
                "custom_only_datasets": [dataset.name for dataset in custom_only.datasets],
                "paper_plus_custom_datasets": len(paper_plus_custom.datasets),
                "manifest_rows_for_demo_item": len(manifest),
                "first_prompt": first_prompt,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
