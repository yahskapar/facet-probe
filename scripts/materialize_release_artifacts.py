#!/usr/bin/env python3
"""Materialize sanitized public artifacts from the paper workspace.

Run from the facet-probe repo root:

    python scripts/materialize_release_artifacts.py

The script is intentionally deterministic and conservative: it copies aggregate
and derived artifacts, plus a sanitized image-set screen that excludes upstream
question text and answer choices.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT
sys.path.insert(0, str(REPO_ROOT / "src"))

from facet_probe.datasets import list_paper_datasets  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_public_text(src: Path, dst: Path) -> None:
    copy_file(src, dst)
    text = dst.read_text(encoding="utf-8")
    text = text.replace("/home/user/health_sensing/", "")
    dst.write_text(text, encoding="utf-8")


def copy_public_artifact(src: Path, dst: Path) -> None:
    if src.suffix == ".parquet":
        copy_file(src, dst)
    else:
        copy_public_text(src, dst)


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def answer_content_key(row: dict[str, Any]) -> str | None:
    """Return an order-invariant answer key when a choice permutation is present."""

    answer_normalized = row.get("answer_normalized")
    if answer_normalized is not None:
        return f"src{answer_normalized}"

    letter = row.get("answer_letter")
    if not isinstance(letter, str) or not letter:
        return None

    sequence = row.get("sequence")
    n_choices = row.get("n_choices", 0)
    if isinstance(sequence, list) and len(sequence) == n_choices:
        idx = ord(letter[0].upper()) - ord("A")
        if 0 <= idx < len(sequence):
            return f"src{sequence[idx]}"
    return letter[0].upper()


def summarize_trial_jsonl(path: Path) -> dict[str, Any]:
    items: set[str] = set()
    answers_by_item: dict[str, set[str]] = defaultdict(set)
    correctness: list[bool] = []
    n_trials = 0
    for row in iter_jsonl(path):
        n_trials += 1
        item_id = str(row["item_id"])
        items.add(item_id)
        key = answer_content_key(row)
        if key is not None:
            answers_by_item[item_id].add(key)
        if row.get("correct") is not None:
            correctness.append(bool(row["correct"]))

    if not items:
        raise ValueError(f"no items found in {path}")
    return {
        "n_items": len(items),
        "n_trials": n_trials,
        "flip_rate": sum(len(answers_by_item[item]) > 1 for item in items) / len(items),
        "accuracy": (sum(correctness) / len(correctness) if correctness else None),
    }


def resolve_result(results_root: Path, glob_pattern: str) -> Path:
    matches = sorted(results_root.glob(glob_pattern))
    if not matches:
        raise FileNotFoundError(f"no result matches {glob_pattern}")
    return matches[0]


def round6(value: float | None) -> float | str:
    if value is None:
        return ""
    return round(value, 6)


def source_ref(source_root: Path, path: Path) -> str:
    return path.relative_to(source_root).as_posix()


def copy_public_markdown(src: Path, dst: Path) -> None:
    copy_public_text(src, dst)
    text = dst.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []
    removed_internal_tasks = False
    skip_section = False

    for line in lines:
        if line.startswith("## Downstream Paper Updates Made"):
            removed_internal_tasks = True
            skip_section = True
            continue
        if line.startswith("Paper target: `reference_akshays_edits/"):
            out.append("Paper target: arXiv appendix ODI methodology section.")
            continue
        if line.startswith("Remaining arXiv-facing work:"):
            removed_internal_tasks = True
            break
        if skip_section and line.startswith("## "):
            skip_section = False
        if skip_section:
            continue
        out.append(line)

    if removed_internal_tasks:
        while out and not out[-1].strip():
            out.pop()
        out.extend(
            [
                "",
                "## Public Release Note",
                "",
                "Internal manuscript-edit task lists were omitted from this public note.",
            ]
        )
    dst.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def materialize_dataset_summary() -> None:
    rows = []
    for spec in list_paper_datasets():
        rows.append(
            {
                "dataset": spec.name,
                "hf_repo": spec.hf_repo,
                "config": spec.config or "",
                "split": spec.split,
                "audited_n": spec.audited_n,
                "license": spec.license,
                "primary_facets": ";".join(spec.primary_facets),
                "notes": spec.notes,
            }
        )
    write_csv(OUTPUT_ROOT / "artifacts/paper/dataset_summary.csv", rows)


def materialize_panel(source_root: Path) -> None:
    src = source_root / "arxiv_finalization/derived/screened_panel_values.json"
    obj = json.loads(src.read_text(encoding="utf-8"))
    write_json(OUTPUT_ROOT / "artifacts/paper/screened_panel_values.json", obj)

    table_rows = []
    for row in obj["table1_rows"]:
        table_rows.append(
            {
                "model": row["model"],
                "display": row["display"],
                "option_order": row["option_order"],
                "evidence_chunk_order": row["evidence_chunk_order"],
                "document_rank_order": row["document_rank_order"],
                "image_set_order_screened": row["image_set_order_screened"],
                "mixed_modality_order": row["mixed_modality_order"],
                "mean5": row["mean5"],
            }
        )
    write_csv(OUTPUT_ROOT / "artifacts/paper/per_facet_per_model.csv", table_rows)

    means = [
        {"facet": facet, "flip_rate": value}
        for facet, value in sorted(obj["facet_means"].items())
    ]
    write_csv(OUTPUT_ROOT / "artifacts/paper/panel_means.csv", means)

    mix_rows = []
    for model, cells in sorted(obj["mixed_modality_cells"].items()):
        for cell in cells:
            mix_rows.append(
                {
                    "model": model,
                    "dataset": cell["dataset"],
                    "sem_flip": cell["sem_flip"],
                    "source": cell.get("source", ""),
                }
            )
    write_csv(OUTPUT_ROOT / "artifacts/paper/mixed_modality_cells.csv", mix_rows)

    image_rows = []
    for model, cells in sorted(obj["image_rates"].items()):
        for dataset, cell in sorted(cells.items()):
            if dataset == "mean":
                continue
            image_rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "k6_flip_rate": cell["k6_flip_rate"],
                    "n_screened": cell["n_screened"],
                    "n_raw_in_file": cell["n_raw_in_file"],
                    "source": cell.get("source", ""),
                }
            )
    write_csv(OUTPUT_ROOT / "artifacts/paper/image_set_screened_rates.csv", image_rows)


def materialize_screen(source_root: Path) -> None:
    src = source_root / "MMIOS/data/screens/imageset_position_reference_screen.json"
    obj = json.loads(src.read_text(encoding="utf-8"))
    summary = {
        "title": obj["header"]["title"],
        "date_built": obj["header"]["date_built"],
        "purpose": obj["header"]["purpose"],
        "classification_rules": obj["header"]["classification_rules"],
        "summary_counts": obj["header"]["summary_counts"],
        "public_sanitization": (
            "Question text, choices, and free-text rationales are omitted here to avoid "
            "redistributing upstream dataset content. Use item IDs with the upstream "
            "datasets to inspect the original records under their licenses."
        ),
    }
    screen_summary = (
        OUTPUT_ROOT / "artifacts/screens/imageset_position_reference_screen_summary.json"
    )
    write_json(screen_summary, summary)

    rows = []
    for dataset, items in obj["items"].items():
        for item in items:
            tracking = item.get("behavioral_tracking") or {}
            accuracy = item.get("behavioral_accuracy") or {}
            rows.append(
                {
                    "dataset": dataset,
                    "item_id": item.get("item_id", ""),
                    "hf_id": item.get("hf_id", ""),
                    "classification": item.get("classification", ""),
                    "k_images": item.get("k_images", ""),
                    "n_choices": item.get("n_choices", ""),
                    "gold_letter": item.get("gold_letter", ""),
                    "regex_flags": ";".join(item.get("regex_flags", [])),
                    "flash_tracking": tracking.get("flash", ""),
                    "pro_tracking": tracking.get("pro", ""),
                    "flash_accuracy": accuracy.get("flash", ""),
                    "pro_accuracy": accuracy.get("pro", ""),
                }
            )
    write_csv(OUTPUT_ROOT / "artifacts/screens/imageset_position_reference_screen.csv", rows)


def materialize_odi(source_root: Path) -> None:
    modal_dir = (
        source_root
        / "MMIOS/results/experiments/"
        / "irt_v6_screened5_modal_2026-06-23_04-42-33_2026-06-23_04-42-34"
    )
    correct_dir = (
        source_root
        / "MMIOS/results/experiments/"
        / "irt_v6_screened5_correct_2026-06-23_04-42-33_2026-06-23_04-42-37"
    )

    modal_summary = json.loads((modal_dir / "irt_v4_modal_per_facet_summary.json").read_text())
    rows = []
    option = modal_summary["facets"]["option_order"]
    for facet, cell in sorted(modal_summary["facets"].items()):
        row = {"facet": facet, **cell}
        row["sigma_ratio_vs_option"] = cell["sigma_pi_median"] / option["sigma_pi_median"]
        if facet == "mixed_modality_order":
            row["delta_ratio_vs_option"] = ""
        else:
            row["delta_ratio_vs_option"] = cell["delta_abs_mean"] / option["delta_abs_mean"]
        rows.append(row)
    write_csv(OUTPUT_ROOT / "artifacts/odi/facet_decomposition.csv", rows)

    posterior_rows = [
        {
            "facet": "option_order",
            "sigma_pi_median": 0.086091,
            "sigma_pi_hdi89_low": 0.050446,
            "sigma_pi_hdi89_high": 0.118783,
            "sigma_delta_median": 0.012044,
            "sigma_delta_hdi89_low": 0.0,
            "sigma_delta_hdi89_high": 0.033048,
        },
        {
            "facet": "document_rank_order",
            "sigma_pi_median": 0.093481,
            "sigma_pi_hdi89_low": 0.053761,
            "sigma_pi_hdi89_high": 0.136924,
            "sigma_delta_median": 0.027487,
            "sigma_delta_hdi89_low": 0.000003,
            "sigma_delta_hdi89_high": 0.062920,
        },
        {
            "facet": "evidence_chunk_order",
            "sigma_pi_median": 0.102758,
            "sigma_pi_hdi89_low": 0.057957,
            "sigma_pi_hdi89_high": 0.146401,
            "sigma_delta_median": 0.027626,
            "sigma_delta_hdi89_low": 0.000056,
            "sigma_delta_hdi89_high": 0.059650,
        },
        {
            "facet": "image_set_order",
            "sigma_pi_median": 0.147333,
            "sigma_pi_hdi89_low": 0.077929,
            "sigma_pi_hdi89_high": 0.224815,
            "sigma_delta_median": 0.063795,
            "sigma_delta_hdi89_low": 0.000011,
            "sigma_delta_hdi89_high": 0.143136,
        },
        {
            "facet": "mixed_modality_order",
            "sigma_pi_median": 0.246045,
            "sigma_pi_hdi89_low": 0.094650,
            "sigma_pi_hdi89_high": 0.415704,
            "sigma_delta_median": 2.363502,
            "sigma_delta_hdi89_low": 1.972798,
            "sigma_delta_hdi89_high": 2.764298,
        },
    ]
    write_csv(OUTPUT_ROOT / "artifacts/odi/posterior_intervals.csv", posterior_rows)

    for filename in [
        "irt_v4_modal_per_facet_summary.json",
        "irt_v4_modal_theta.json",
        "irt_v4_modal_diagnostics.json",
        "irt_input_summary.json",
        "screen_filter_metadata.json",
        "irt_v4_modal_per_item_params.parquet",
    ]:
        copy_public_artifact(modal_dir / filename, OUTPUT_ROOT / "artifacts/odi" / filename)
    for filename in [
        "irt_v4_correct_per_facet_summary.json",
        "irt_v4_correct_theta.json",
        "irt_v4_correct_diagnostics.json",
        "irt_v4_correct_per_item_params.parquet",
    ]:
        copy_public_artifact(correct_dir / filename, OUTPUT_ROOT / "artifacts/odi" / filename)


def materialize_robustness(source_root: Path) -> None:
    for filename in ["decoder_decomp_screened_by_facet.csv", "decoder_decomp_screened.json"]:
        copy_public_text(
            source_root / "MMIOS/results" / filename,
            OUTPUT_ROOT / "artifacts/robustness" / filename,
        )
    copy_public_markdown(
        source_root / "arxiv_finalization/derived/decoder_decomp_screened_audit.md",
        OUTPUT_ROOT / "artifacts/provenance/decoder_decomp_screened_audit.md",
    )


def materialize_mitigation(source_root: Path) -> None:
    src = source_root / "arxiv_finalization/derived/mitigation_screened_values.json"
    obj = json.loads(src.read_text(encoding="utf-8"))
    write_json(OUTPUT_ROOT / "artifacts/mitigation/mitigation_screened_values.json", obj)
    rows = []
    for key, cell in obj["policy_means"].items():
        rows.append({"policy_id": key, **cell})
    write_csv(OUTPUT_ROOT / "artifacts/mitigation/policy_means.csv", rows)

    exp_dir = source_root / "MMIOS/results/experiments/mitigation_menu_n200_2026-05-11_23-15-45"
    for filename in ["mitigation_menu_summary.json", "pareto_frontier_points.json"]:
        copy_public_text(exp_dir / filename, OUTPUT_ROOT / "artifacts/mitigation" / filename)
    copy_public_markdown(
        source_root / "arxiv_finalization/derived/mitigation_screened_values.md",
        OUTPUT_ROOT / "artifacts/provenance/mitigation_screened_values.md",
    )


def materialize_prompt_mitigation(source_root: Path) -> None:
    results_root = source_root / "MMIOS/results/experiments"
    cta_cells = [
        {
            "dataset": "MedXpertQA",
            "dataset_id": "medxpertqa",
            "modality": "text-evidence",
            "model": "gemini-3.1-pro-preview",
            "model_display": "Gemini 3.1 Pro",
            "baseline": (
                "tf_full_2026-05-21*/inference/"
                "medxpertqa__gemini-3.1-pro-preview__baseline_think8k.jsonl"
            ),
            "cta": (
                "tf_full_2026-05-21*/inference/"
                "medxpertqa__gemini-3.1-pro-preview__cta.jsonl"
            ),
            "cta_multipass": (
                "tfx_full_n100_2026-05-21*/inference/"
                "medxpertqa__gemini-3.1-pro-preview__cta_multipass.jsonl"
            ),
        },
        {
            "dataset": "MedXpertQA",
            "dataset_id": "medxpertqa",
            "modality": "text-evidence",
            "model": "gemini-3-flash-preview",
            "model_display": "Gemini 3 Flash",
            "baseline": (
                "tf_full_2026-05-21*/inference/"
                "medxpertqa__gemini-3-flash-preview__baseline_think8k.jsonl"
            ),
            "cta": (
                "tf_full_2026-05-21*/inference/"
                "medxpertqa__gemini-3-flash-preview__cta.jsonl"
            ),
            "cta_multipass": (
                "tfx_full_n100_2026-05-21*/inference/"
                "medxpertqa__gemini-3-flash-preview__cta_multipass.jsonl"
            ),
        },
        {
            "dataset": "MMLU-Pro",
            "dataset_id": "mmlu_pro",
            "modality": "text-mcq",
            "model": "gemini-3.1-pro-preview",
            "model_display": "Gemini 3.1 Pro",
            "baseline": (
                "tfx_full_n100_2026-05-21*/inference/"
                "mmlu_pro__gemini-3.1-pro-preview__baseline_think8k.jsonl"
            ),
            "cta": (
                "tfx_full_n100_2026-05-21*/inference/"
                "mmlu_pro__gemini-3.1-pro-preview__cta.jsonl"
            ),
        },
        {
            "dataset": "MMLU-Pro",
            "dataset_id": "mmlu_pro",
            "modality": "text-mcq",
            "model": "gemini-3-flash-preview",
            "model_display": "Gemini 3 Flash",
            "baseline": (
                "tfx_full_n100_2026-05-21*/inference/"
                "mmlu_pro__gemini-3-flash-preview__baseline_think8k.jsonl"
            ),
            "cta": (
                "tfx_full_n100_2026-05-21*/inference/"
                "mmlu_pro__gemini-3-flash-preview__cta.jsonl"
            ),
        },
        {
            "dataset": "CSQA",
            "dataset_id": "commonsenseqa",
            "modality": "text-mcq",
            "model": "gemini-3.1-pro-preview",
            "model_display": "Gemini 3.1 Pro",
            "baseline": (
                "tfx_full_n100_2026-05-21*/inference/"
                "commonsenseqa__gemini-3.1-pro-preview__baseline_think8k.jsonl"
            ),
            "cta": (
                "tfx_full_n100_2026-05-21*/inference/"
                "commonsenseqa__gemini-3.1-pro-preview__cta.jsonl"
            ),
        },
        {
            "dataset": "CSQA",
            "dataset_id": "commonsenseqa",
            "modality": "text-mcq",
            "model": "gemini-3-flash-preview",
            "model_display": "Gemini 3 Flash",
            "baseline": (
                "tfx_full_n100_2026-05-21*/inference/"
                "commonsenseqa__gemini-3-flash-preview__baseline_think8k.jsonl"
            ),
            "cta": (
                "tfx_full_n100_2026-05-21*/inference/"
                "commonsenseqa__gemini-3-flash-preview__cta.jsonl"
            ),
        },
        {
            "dataset": "MathVision",
            "dataset_id": "mathvision",
            "modality": "visual-mcq",
            "model": "gemini-3.1-pro-preview",
            "model_display": "Gemini 3.1 Pro",
            "baseline": (
                "wwe_full_2026-05-22*/inference/"
                "mathvision__gemini-3.1-pro-preview__baseline_think8k.jsonl"
            ),
            "cta": (
                "wwe_full_2026-05-22*/inference/"
                "mathvision__gemini-3.1-pro-preview__cta.jsonl"
            ),
        },
        {
            "dataset": "MathVision",
            "dataset_id": "mathvision",
            "modality": "visual-mcq",
            "model": "gemini-3-flash-preview",
            "model_display": "Gemini 3 Flash",
            "baseline": (
                "wwe_full_2026-05-22*/inference/"
                "mathvision__gemini-3-flash-preview__baseline_think8k.jsonl"
            ),
            "cta": (
                "wwe_full_2026-05-22*/inference/"
                "mathvision__gemini-3-flash-preview__cta.jsonl"
            ),
        },
    ]

    cta_rows: list[dict[str, Any]] = []
    for cell in cta_cells:
        baseline_path = resolve_result(results_root, cell["baseline"])
        cta_path = resolve_result(results_root, cell["cta"])
        baseline = summarize_trial_jsonl(baseline_path)
        cta = summarize_trial_jsonl(cta_path)
        delta_abs = cta["flip_rate"] - baseline["flip_rate"]
        delta_relative_pct = (
            100 * delta_abs / baseline["flip_rate"]
            if baseline["flip_rate"] > 0
            else None
        )
        row = {
            "dataset": cell["dataset"],
            "dataset_id": cell["dataset_id"],
            "modality": cell["modality"],
            "model": cell["model"],
            "model_display": cell["model_display"],
            "baseline_flip_rate": round6(baseline["flip_rate"]),
            "cta_flip_rate": round6(cta["flip_rate"]),
            "delta_abs": round6(delta_abs),
            "delta_relative_pct": round6(delta_relative_pct),
            "baseline_accuracy": round6(baseline["accuracy"]),
            "cta_accuracy": round6(cta["accuracy"]),
            "n_items_baseline": baseline["n_items"],
            "n_items_cta": cta["n_items"],
            "n_trials_baseline": baseline["n_trials"],
            "n_trials_cta": cta["n_trials"],
            "baseline_source": source_ref(source_root, baseline_path),
            "cta_source": source_ref(source_root, cta_path),
            "cta_multipass_flip_rate": "",
            "cta_multipass_delta_vs_cta": "",
            "cta_multipass_accuracy": "",
            "n_items_cta_multipass": "",
            "cta_multipass_source": "",
        }
        if cell.get("cta_multipass"):
            multipass_path = resolve_result(results_root, cell["cta_multipass"])
            multipass = summarize_trial_jsonl(multipass_path)
            row.update(
                cta_multipass_flip_rate=round6(multipass["flip_rate"]),
                cta_multipass_delta_vs_cta=round6(multipass["flip_rate"] - cta["flip_rate"]),
                cta_multipass_accuracy=round6(multipass["accuracy"]),
                n_items_cta_multipass=multipass["n_items"],
                cta_multipass_source=source_ref(source_root, multipass_path),
            )
        cta_rows.append(row)
    write_csv(OUTPUT_ROOT / "artifacts/mitigation/cta_flip_summary.csv", cta_rows)

    think_cells = [
        (
            "MedXpertQA",
            "medxpertqa",
            "hard",
            "evidence_chunk_order",
            "gemini-3.1-pro-preview",
            "Gemini 3.1 Pro",
            1024,
            "1k",
            "think_1k",
            (
                "tfx_full_n100_2026-05-21*/inference/"
                "medxpertqa__gemini-3.1-pro-preview__think_1k.jsonl"
            ),
        ),
        (
            "MedXpertQA",
            "medxpertqa",
            "hard",
            "evidence_chunk_order",
            "gemini-3.1-pro-preview",
            "Gemini 3.1 Pro",
            2048,
            "2k",
            "think_2k",
            (
                "tf_full_2026-05-21*/inference/"
                "medxpertqa__gemini-3.1-pro-preview__think_2k.jsonl"
            ),
        ),
        (
            "MedXpertQA",
            "medxpertqa",
            "hard",
            "evidence_chunk_order",
            "gemini-3.1-pro-preview",
            "Gemini 3.1 Pro",
            8192,
            "8k",
            "baseline_think8k",
            (
                "tf_full_2026-05-21*/inference/"
                "medxpertqa__gemini-3.1-pro-preview__baseline_think8k.jsonl"
            ),
        ),
        (
            "MedXpertQA",
            "medxpertqa",
            "hard",
            "evidence_chunk_order",
            "gemini-3.1-pro-preview",
            "Gemini 3.1 Pro",
            24576,
            "24k",
            "think_24k",
            (
                "tf_full_2026-05-21*/inference/"
                "medxpertqa__gemini-3.1-pro-preview__think_24k.jsonl"
            ),
        ),
        (
            "MedXpertQA",
            "medxpertqa",
            "hard",
            "evidence_chunk_order",
            "gemini-3-flash-preview",
            "Gemini 3 Flash",
            1024,
            "1k",
            "think_1k",
            (
                "tfx_full_n100_2026-05-21*/inference/"
                "medxpertqa__gemini-3-flash-preview__think_1k.jsonl"
            ),
        ),
        (
            "MedXpertQA",
            "medxpertqa",
            "hard",
            "evidence_chunk_order",
            "gemini-3-flash-preview",
            "Gemini 3 Flash",
            2048,
            "2k",
            "think_2k",
            (
                "tb_medx_flash_2026-05-25*/inference/"
                "medxpertqa__gemini-3-flash-preview__think_2k.jsonl"
            ),
        ),
        (
            "MedXpertQA",
            "medxpertqa",
            "hard",
            "evidence_chunk_order",
            "gemini-3-flash-preview",
            "Gemini 3 Flash",
            8192,
            "8k",
            "baseline_think8k",
            (
                "tf_full_2026-05-21*/inference/"
                "medxpertqa__gemini-3-flash-preview__baseline_think8k.jsonl"
            ),
        ),
        (
            "MedXpertQA",
            "medxpertqa",
            "hard",
            "evidence_chunk_order",
            "gemini-3-flash-preview",
            "Gemini 3 Flash",
            24576,
            "24k",
            "think_24k",
            (
                "tb_medx_flash_2026-05-25*/inference/"
                "medxpertqa__gemini-3-flash-preview__think_24k.jsonl"
            ),
        ),
        (
            "MMLU-Pro",
            "mmlu_pro",
            "easy",
            "option_order",
            "gemini-3.1-pro-preview",
            "Gemini 3.1 Pro",
            1024,
            "1k",
            "think_1k",
            (
                "tb_mmlu_2026-05-25*/inference/"
                "mmlu_pro__gemini-3.1-pro-preview__think_1k.jsonl"
            ),
        ),
        (
            "MMLU-Pro",
            "mmlu_pro",
            "easy",
            "option_order",
            "gemini-3.1-pro-preview",
            "Gemini 3.1 Pro",
            2048,
            "2k",
            "think_2k",
            (
                "tb_mmlu_2026-05-25*/inference/"
                "mmlu_pro__gemini-3.1-pro-preview__think_2k.jsonl"
            ),
        ),
        (
            "MMLU-Pro",
            "mmlu_pro",
            "easy",
            "option_order",
            "gemini-3.1-pro-preview",
            "Gemini 3.1 Pro",
            8192,
            "8k",
            "baseline_think8k",
            (
                "tb_mmlu_2026-05-25*/inference/"
                "mmlu_pro__gemini-3.1-pro-preview__baseline_think8k.jsonl"
            ),
        ),
        (
            "MMLU-Pro",
            "mmlu_pro",
            "easy",
            "option_order",
            "gemini-3.1-pro-preview",
            "Gemini 3.1 Pro",
            24576,
            "24k",
            "think_24k",
            (
                "tb_mmlu_2026-05-25*/inference/"
                "mmlu_pro__gemini-3.1-pro-preview__think_24k.jsonl"
            ),
        ),
        (
            "MMLU-Pro",
            "mmlu_pro",
            "easy",
            "option_order",
            "gemini-3-flash-preview",
            "Gemini 3 Flash",
            1024,
            "1k",
            "think_1k",
            (
                "tb_mmlu_2026-05-25*/inference/"
                "mmlu_pro__gemini-3-flash-preview__think_1k.jsonl"
            ),
        ),
        (
            "MMLU-Pro",
            "mmlu_pro",
            "easy",
            "option_order",
            "gemini-3-flash-preview",
            "Gemini 3 Flash",
            2048,
            "2k",
            "think_2k",
            (
                "tb_mmlu_2026-05-25*/inference/"
                "mmlu_pro__gemini-3-flash-preview__think_2k.jsonl"
            ),
        ),
        (
            "MMLU-Pro",
            "mmlu_pro",
            "easy",
            "option_order",
            "gemini-3-flash-preview",
            "Gemini 3 Flash",
            8192,
            "8k",
            "baseline_think8k",
            (
                "tb_mmlu_2026-05-25*/inference/"
                "mmlu_pro__gemini-3-flash-preview__baseline_think8k.jsonl"
            ),
        ),
        (
            "MMLU-Pro",
            "mmlu_pro",
            "easy",
            "option_order",
            "gemini-3-flash-preview",
            "Gemini 3 Flash",
            24576,
            "24k",
            "think_24k",
            (
                "tb_mmlu_2026-05-25*/inference/"
                "mmlu_pro__gemini-3-flash-preview__think_24k.jsonl"
            ),
        ),
    ]

    think_rows = []
    for (
        dataset,
        dataset_id,
        regime,
        facet,
        model,
        model_display,
        budget,
        budget_label,
        intervention,
        pattern,
    ) in think_cells:
        path = resolve_result(results_root, pattern)
        summary = summarize_trial_jsonl(path)
        think_rows.append(
            {
                "dataset": dataset,
                "dataset_id": dataset_id,
                "regime": regime,
                "facet": facet,
                "model": model,
                "model_display": model_display,
                "thinking_budget_tokens": budget,
                "budget_label": budget_label,
                "intervention": intervention,
                "flip_rate": round6(summary["flip_rate"]),
                "accuracy": round6(summary["accuracy"]),
                "n_items": summary["n_items"],
                "n_trials": summary["n_trials"],
                "source": source_ref(source_root, path),
            }
        )
    write_csv(OUTPUT_ROOT / "artifacts/mitigation/think_budget_sweep.csv", think_rows)

    provenance = """# CTA and Think-Budget Summary Provenance

These compact public summaries back the Figure 4 CTA and think-budget claims
without redistributing upstream dataset content or raw provider outputs.

Generated by `scripts/materialize_release_artifacts.py` from the paper
workspace JSONL result files under `MMIOS/results/experiments/`.

Flip-rate rule:

- use `answer_normalized` when present;
- otherwise map answer letters back through `sequence` when the sequence is an
  option permutation;
- otherwise use answer letters directly, which is the evidence-order case for
  MedXpertQA.

`cta_flip_summary.csv` reports baseline-vs-CTA rows, plus CTA+multi-pass rows
for the MedXpertQA anti-synergy claim. `think_budget_sweep.csv` reports the
hard MedXpertQA and easy MMLU-Pro budget sweeps used in the figure.

The paper's uncertainty-clean MedXpertQA subset is discussed in the manuscript,
but the item-level subset labels and raw model-output JSONL are not shipped in
this initial public artifact release.
"""
    (OUTPUT_ROOT / "artifacts/provenance").mkdir(parents=True, exist_ok=True)
    (
        OUTPUT_ROOT / "artifacts/provenance/cta_think_budget_summary.md"
    ).write_text(provenance, encoding="utf-8")


def materialize_provenance(source_root: Path) -> None:
    for rel in [
        "arxiv_finalization/derived/screened_panel_values.md",
        "arxiv_finalization/derived/screened_odi_fit_audit.md",
        "arxiv_finalization/derived/odi_posterior_interval_audit.md",
    ]:
        dst = OUTPUT_ROOT / "artifacts/provenance" / Path(rel).name
        copy_public_markdown(source_root / rel, dst)


def main() -> int:
    global OUTPUT_ROOT

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT.parent,
        help="Path to EMNLP_2026 paper workspace (default: parent of this repo).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT,
        help="Directory under which artifacts/ will be written (default: repo root).",
    )
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    if not (source_root / "MMIOS").exists():
        parser.error(f"source root does not look like EMNLP_2026: {source_root}")
    OUTPUT_ROOT = args.output_root.resolve()

    materialize_dataset_summary()
    materialize_panel(source_root)
    materialize_screen(source_root)
    materialize_odi(source_root)
    materialize_robustness(source_root)
    materialize_mitigation(source_root)
    materialize_prompt_mitigation(source_root)
    materialize_provenance(source_root)
    print(f"materialized release artifacts under {OUTPUT_ROOT / 'artifacts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
