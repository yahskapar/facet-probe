#!/usr/bin/env python3
"""Audit the public Facet-Probe release tree.

The audit is intentionally offline and deterministic. It checks that every file
under artifacts/ is manifest-listed, that materialized artifacts can be
regenerated from the local paper workspace, that aggregate values pass the
package verifier, that sanitized screens do not expose upstream question
content, and that no obvious credentials are present in text files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from facet_probe.artifacts import verify_release_artifacts  # noqa: E402


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "assigned secret",
        re.compile(
            r"\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*[\"'][^\"']{8,}[\"']",
            re.IGNORECASE,
        ),
    ),
)

SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
SKIP_SUFFIXES = {".parquet", ".pyc", ".pyo", ".png", ".pdf", ".zip"}


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, Any]:
    path = REPO_ROOT / "configs/release_artifacts.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def round_half_up(value: str | float, digits: int) -> float:
    quant = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def flatten_manifest_entries(value: Any) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            out.extend(flatten_manifest_entries(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(flatten_manifest_entries(item))
    return out


def check_manifest_coverage(manifest: dict[str, Any]) -> list[Check]:
    expected = set(flatten_manifest_entries(manifest["included"]))
    actual = {rel(path) for path in (REPO_ROOT / "artifacts").rglob("*") if path.is_file()}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    checks = [
        Check(
            "manifest includes every shipped artifact",
            not extra,
            "extra=" + (", ".join(extra) if extra else "none"),
        ),
        Check(
            "manifest-listed artifacts exist",
            not missing,
            "missing=" + (", ".join(missing) if missing else "none"),
        ),
    ]

    correspondence_paths = set()
    for item in manifest.get("correspondence", []):
        correspondence_paths.update(item.get("artifacts", []))
    uncovered = sorted(expected - correspondence_paths - {"artifacts/README.md"})
    checks.append(
        Check(
            "artifact correspondences cover shipped data artifacts",
            not uncovered,
            "uncovered=" + (", ".join(uncovered) if uncovered else "none"),
        )
    )
    return checks


def check_config_artifact_coverage() -> list[Check]:
    datasets = yaml.safe_load((REPO_ROOT / "configs/datasets.yaml").read_text(encoding="utf-8"))
    models = yaml.safe_load((REPO_ROOT / "configs/models.yaml").read_text(encoding="utf-8"))
    facets = yaml.safe_load((REPO_ROOT / "configs/facets.yaml").read_text(encoding="utf-8"))

    dataset_rows = load_csv_rows(REPO_ROOT / "artifacts/paper/dataset_summary.csv")
    per_model_rows = load_csv_rows(REPO_ROOT / "artifacts/paper/per_facet_per_model.csv")
    panel_rows = load_csv_rows(REPO_ROOT / "artifacts/paper/panel_means.csv")

    dataset_expected = set(datasets.get("datasets", {}))
    dataset_actual = {row["dataset"] for row in dataset_rows}

    model_expected = {spec["display"] for spec in models.get("models", {}).values()}
    model_actual = {row["display"] for row in per_model_rows}

    main_facets = {
        name
        for name, spec in facets.get("facets", {}).items()
        if spec.get("status") == "main"
    }
    artifact_facets = {row["facet"] for row in panel_rows if row["facet"] != "mean5"}
    artifact_facets_normalized = {
        "image_set_order" if facet == "image_set_order_screened" else facet
        for facet in artifact_facets
    }

    return [
        Check(
            "dataset config covers artifact dataset summary",
            dataset_expected == dataset_actual,
            "missing="
            + (", ".join(sorted(dataset_expected - dataset_actual)) or "none")
            + "; extra="
            + (", ".join(sorted(dataset_actual - dataset_expected)) or "none"),
        ),
        Check(
            "model config covers per-model artifact rows",
            model_expected == model_actual,
            "missing="
            + (", ".join(sorted(model_expected - model_actual)) or "none")
            + "; extra="
            + (", ".join(sorted(model_actual - model_expected)) or "none"),
        ),
        Check(
            "main facet config covers panel artifact facets",
            main_facets == artifact_facets_normalized,
            "missing="
            + (", ".join(sorted(main_facets - artifact_facets_normalized)) or "none")
            + "; extra="
            + (", ".join(sorted(artifact_facets_normalized - main_facets)) or "none"),
        ),
    ]


def check_materialized_reproducibility(
    manifest: dict[str, Any],
    source_root: Path | None,
    *,
    offline_only: bool = False,
) -> list[Check]:
    if offline_only:
        return [
            Check(
                "materialized artifacts reproduce from paper workspace",
                True,
                "skipped by --offline-only",
            )
        ]
    if source_root is None:
        default_root = REPO_ROOT.parent
        source_root = default_root if (default_root / "MMIOS").exists() else None
    if source_root is None:
        return [
            Check(
                "materialized artifacts reproduce from paper workspace",
                False,
                "source root not found; pass --source-root /path/to/EMNLP_2026",
            )
        ]

    materialized = list(manifest.get("materialized_by_script", []))
    with tempfile.TemporaryDirectory(prefix="facet_probe_release_audit_") as tmp:
        tmp_root = Path(tmp)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/materialize_release_artifacts.py"),
            "--source-root",
            str(source_root),
            "--output-root",
            str(tmp_root),
        ]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
        if proc.returncode != 0:
            return [
                Check(
                    "materialized artifacts reproduce from paper workspace",
                    False,
                    proc.stderr.strip() or proc.stdout.strip(),
                )
            ]

        mismatches = []
        for rel_path in materialized:
            current = REPO_ROOT / rel_path
            regenerated = tmp_root / rel_path
            if not regenerated.exists():
                mismatches.append(f"{rel_path}: not regenerated")
            elif sha256(current) != sha256(regenerated):
                mismatches.append(f"{rel_path}: sha256 mismatch")
        return [
            Check(
                "materialized artifacts reproduce from paper workspace",
                not mismatches,
                "mismatches=" + (", ".join(mismatches) if mismatches else "none"),
            )
        ]


def strip_tex_cell(text: str) -> str:
    text = re.sub(r"\\(?:textbf|underline|emph|boldsymbol)\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\\[A-Za-z]+\{?\}?", "", text)
    return text.strip()


def parse_arxiv_table1(findings_tex: str) -> dict[str, list[float]]:
    match = re.search(
        r"Model & .*?\\\\\s*\\midrule(.*?)\\midrule\s*\\emph\{Panel mean",
        findings_tex,
        re.DOTALL,
    )
    if match is None:
        return {}
    rows: dict[str, list[float]] = {}
    for line in match.group(1).splitlines():
        if "&" not in line or "\\\\" not in line:
            continue
        if line.strip().startswith("%") or "midrule" in line:
            continue
        parts = [strip_tex_cell(part.replace("\\\\", "")) for part in line.split("&")]
        if len(parts) == 7:
            rows[parts[0]] = [float(part) for part in parts[1:]]
    return rows


def check_arxiv_table1_alignment(findings_tex: str) -> Check:
    tex_rows = parse_arxiv_table1(findings_tex)
    artifact_rows = load_csv_rows(REPO_ROOT / "artifacts/paper/per_facet_per_model.csv")
    columns = [
        "option_order",
        "evidence_chunk_order",
        "document_rank_order",
        "image_set_order_screened",
        "mixed_modality_order",
        "mean5",
    ]
    mismatches = []
    for row in artifact_rows:
        name = row["display"]
        rounded = [round_half_up(row[column], 2) for column in columns]
        if tex_rows.get(name) != rounded:
            mismatches.append(f"{name}: artifact={rounded} tex={tex_rows.get(name)}")
    artifact_names = {row["display"] for row in artifact_rows}
    tex_names = set(tex_rows)
    missing = sorted(artifact_names - tex_names)
    extra = sorted(tex_names - artifact_names)
    detail = (
        "mismatches="
        + ("; ".join(mismatches[:5]) if mismatches else "none")
        + f"; missing={missing or 'none'}; extra={extra or 'none'}"
    )
    return Check(
        "arXiv Table 1 matches per-model artifact values",
        not mismatches and not missing and not extra and len(artifact_rows) == 18,
        detail,
    )


def check_arxiv_dataset_alignment(
    dataset_tex: str,
    methods_tex: str,
    limitations_tex: str,
) -> Check:
    rows = load_csv_rows(REPO_ROOT / "artifacts/paper/dataset_summary.csv")
    expected_n = {
        "mmlu_pro": "200",
        "commonsenseqa": "200",
        "mathvision": "190",
        "hotpotqa": "199",
        "musique": "200",
        "medxpertqa": "150",
        "multihop_rag": "171",
        "mantis_eval": "70 raw / 18 clean",
        "medframeqa": "200 raw / 195 clean",
        "mramg": "197",
        "mmdocrag": "200",
        "mmqa": "200",
    }
    by_name = {row["dataset"]: row for row in rows}
    mismatches = [
        f"{name}: artifact={by_name.get(name, {}).get('audited_n')} tex={expected}"
        for name, expected in expected_n.items()
        if by_name.get(name, {}).get("audited_n") != expected
    ]
    required_phrases = [
        "70 (18 clean)",
        "200 (195 clean)",
        "N{=}597",
        "52/70 Mantis-Eval items excluded; 5/200",
        "18 Mantis-Eval and 195 MedFrameQA items",
    ]
    paper_text = "\n".join([dataset_tex, methods_tex, limitations_tex])
    missing_phrases = [phrase for phrase in required_phrases if phrase not in paper_text]
    return Check(
        "arXiv dataset coverage matches dataset artifact",
        len(rows) == 12 and not mismatches and not missing_phrases,
        "rows="
        + str(len(rows))
        + "; mismatches="
        + ("; ".join(mismatches) if mismatches else "none")
        + "; missing_phrases="
        + (", ".join(missing_phrases) if missing_phrases else "none"),
    )


def check_arxiv_odi_alignment() -> Check:
    rows = {
        row["facet"]: row
        for row in load_csv_rows(REPO_ROOT / "artifacts/odi/facet_decomposition.csv")
    }
    expected = {
        "option_order": (0.088, 1.00, 0.003, 1.00),
        "document_rank_order": (0.097, 1.10, 0.010, 3.81),
        "evidence_chunk_order": (0.105, 1.20, 0.012, 4.50),
        "image_set_order": (0.152, 1.73, 0.022, 8.25),
        "mixed_modality_order": (0.265, 3.03, 4.796, None),
    }
    mismatches = []
    for facet, expected_values in expected.items():
        row = rows.get(facet)
        if row is None:
            mismatches.append(f"{facet}: missing")
            continue
        values = (
            round_half_up(row["sigma_pi_median"], 3),
            round_half_up(row["sigma_ratio_vs_option"], 2),
            round_half_up(row["delta_abs_mean"], 3),
        )
        if values != expected_values[:3]:
            mismatches.append(f"{facet}: artifact={values} tex={expected_values[:3]}")
        if expected_values[3] is not None:
            delta_ratio = round_half_up(row["delta_ratio_vs_option"], 2)
            if delta_ratio != expected_values[3]:
                mismatches.append(f"{facet}: delta_ratio={delta_ratio} tex={expected_values[3]}")
    return Check(
        "arXiv Table 2 matches ODI artifact values",
        len(rows) == 5 and not mismatches,
        "mismatches=" + ("; ".join(mismatches) if mismatches else "none"),
    )


def check_arxiv_posterior_interval_alignment() -> Check:
    rows = {
        row["facet"]: row
        for row in load_csv_rows(REPO_ROOT / "artifacts/odi/posterior_intervals.csv")
    }
    expected = {
        "option_order": (0.086, 0.050, 0.119, 0.012, 0.000, 0.033),
        "document_rank_order": (0.093, 0.054, 0.137, 0.027, 0.000, 0.063),
        "evidence_chunk_order": (0.103, 0.058, 0.146, 0.028, 0.000, 0.060),
        "image_set_order": (0.147, 0.078, 0.225, 0.064, 0.000, 0.143),
        "mixed_modality_order": (0.246, 0.095, 0.416, 2.364, 1.973, 2.764),
    }
    columns = [
        "sigma_pi_median",
        "sigma_pi_hdi89_low",
        "sigma_pi_hdi89_high",
        "sigma_delta_median",
        "sigma_delta_hdi89_low",
        "sigma_delta_hdi89_high",
    ]
    mismatches = []
    for facet, expected_values in expected.items():
        row = rows.get(facet)
        if row is None:
            mismatches.append(f"{facet}: missing")
            continue
        values = tuple(round_half_up(row[column], 3) for column in columns)
        if values != expected_values:
            mismatches.append(f"{facet}: artifact={values} tex={expected_values}")
    return Check(
        "arXiv appendix posterior intervals match artifact values",
        len(rows) == 5 and not mismatches,
        "mismatches=" + ("; ".join(mismatches) if mismatches else "none"),
    )


def check_arxiv_mitigation_alignment() -> Check:
    rows = {
        row["label"].split()[0]: row
        for row in load_csv_rows(REPO_ROOT / "artifacts/mitigation/policy_means.csv")
    }
    expected = {
        "P1": (1, 0.629, 0.000),
        "P2": (2, 0.648, 0.019),
        "P3": (3, 0.648, 0.019),
        "P4": (6, 0.640, 0.011),
        "P5": (6, 0.580, -0.049),
        "P6": (6, 0.674, 0.046),
    }
    mismatches = []
    for policy, expected_values in expected.items():
        row = rows.get(policy)
        if row is None:
            mismatches.append(f"{policy}: missing")
            continue
        values = (
            int(float(row["cost"])),
            round_half_up(row["mean_accuracy"], 3),
            round_half_up(row["delta_vs_p1"], 3),
        )
        if values != expected_values:
            mismatches.append(f"{policy}: artifact={values} tex={expected_values}")
    p2 = rows.get("P2")
    if p2 is not None and round_half_up(p2["coverage"], 3) != 0.797:
        mismatches.append(f"P2 coverage={round_half_up(p2['coverage'], 3)} tex=0.797")
    return Check(
        "arXiv Table 4 matches mitigation artifact values",
        len(rows) == 6 and not mismatches,
        "mismatches=" + ("; ".join(mismatches) if mismatches else "none"),
    )


def check_arxiv_prompt_mitigation_alignment(findings_tex: str, supplement_tex: str) -> Check:
    cta_rows = {
        (row["dataset_id"], row["model"]): row
        for row in load_csv_rows(REPO_ROOT / "artifacts/mitigation/cta_flip_summary.csv")
    }
    expected_cta = {
        ("medxpertqa", "gemini-3.1-pro-preview"): (0.30, 0.18),
        ("medxpertqa", "gemini-3-flash-preview"): (0.26, 0.30),
        ("mmlu_pro", "gemini-3.1-pro-preview"): (0.10, 0.09),
        ("commonsenseqa", "gemini-3.1-pro-preview"): (0.06, 0.07),
        ("mathvision", "gemini-3.1-pro-preview"): (0.29, 0.29),
        ("mathvision", "gemini-3-flash-preview"): (0.35, 0.35),
    }
    mismatches = []
    for key, expected_values in expected_cta.items():
        row = cta_rows.get(key)
        if row is None:
            mismatches.append(f"CTA {key}: missing")
            continue
        values = (
            round_half_up(row["baseline_flip_rate"], 2),
            round_half_up(row["cta_flip_rate"], 2),
        )
        if values != expected_values:
            mismatches.append(f"CTA {key}: artifact={values} tex={expected_values}")

    for key, expected_value in {
        ("medxpertqa", "gemini-3.1-pro-preview"): 0.29,
        ("medxpertqa", "gemini-3-flash-preview"): 0.40,
    }.items():
        row = cta_rows.get(key)
        if row is None:
            continue
        value = round_half_up(row["cta_multipass_flip_rate"], 2)
        if value != expected_value:
            mismatches.append(f"CTA+multi-pass {key}: artifact={value} tex={expected_value}")

    budget_rows = {
        (row["dataset_id"], row["model"], row["budget_label"]): row
        for row in load_csv_rows(REPO_ROOT / "artifacts/mitigation/think_budget_sweep.csv")
    }
    expected_budget = {
        ("medxpertqa", "gemini-3.1-pro-preview", "1k"): 0.41,
        ("medxpertqa", "gemini-3.1-pro-preview", "2k"): 0.32,
        ("medxpertqa", "gemini-3.1-pro-preview", "8k"): 0.30,
        ("medxpertqa", "gemini-3.1-pro-preview", "24k"): 0.28,
        ("medxpertqa", "gemini-3-flash-preview", "1k"): 0.41,
        ("medxpertqa", "gemini-3-flash-preview", "2k"): 0.41,
        ("medxpertqa", "gemini-3-flash-preview", "8k"): 0.26,
        ("medxpertqa", "gemini-3-flash-preview", "24k"): 0.35,
        ("mmlu_pro", "gemini-3.1-pro-preview", "1k"): 0.11,
        ("mmlu_pro", "gemini-3.1-pro-preview", "2k"): 0.10,
        ("mmlu_pro", "gemini-3.1-pro-preview", "8k"): 0.11,
        ("mmlu_pro", "gemini-3.1-pro-preview", "24k"): 0.11,
        ("mmlu_pro", "gemini-3-flash-preview", "1k"): 0.12,
        ("mmlu_pro", "gemini-3-flash-preview", "2k"): 0.10,
        ("mmlu_pro", "gemini-3-flash-preview", "8k"): 0.11,
        ("mmlu_pro", "gemini-3-flash-preview", "24k"): 0.08,
    }
    for key, expected_value in expected_budget.items():
        row = budget_rows.get(key)
        if row is None:
            mismatches.append(f"think-budget {key}: missing")
            continue
        value = round_half_up(row["flip_rate"], 2)
        if value != expected_value:
            mismatches.append(f"think-budget {key}: artifact={value} tex={expected_value}")

    mmlu_pro_acc = [
        float(row["accuracy"])
        for key, row in budget_rows.items()
        if key[0] == "mmlu_pro" and key[1] == "gemini-3.1-pro-preview"
    ]
    mmlu_flash_acc = [
        float(row["accuracy"])
        for key, row in budget_rows.items()
        if key[0] == "mmlu_pro" and key[1] == "gemini-3-flash-preview"
    ]
    if (
        round_half_up(min(mmlu_pro_acc), 3),
        round_half_up(max(mmlu_pro_acc), 3),
    ) != (0.927, 0.938):
        mismatches.append("MMLU-Pro Pro accuracy range mismatch")
    if (
        round_half_up(min(mmlu_flash_acc), 3),
        round_half_up(max(mmlu_flash_acc), 3),
    ) != (0.910, 0.932):
        mismatches.append("MMLU-Pro Flash accuracy range mismatch")

    compact_tex = re.sub(r"\s+", "", findings_tex + supplement_tex)
    required_phrases = [
        r"0.30\to0.18",
        r"0.135\to0.029",
        r"0.26\to0.30",
        r"0.10\to0.09",
        r"0.06\to0.07",
        r"0.29\to0.29",
        r"0.35\to0.35",
        r"0.18\to0.29",
        r"0.30\to0.40",
        r"0.41\to0.28",
        "0.110/0.100/0.110/0.110",
        "0.120/0.100/0.110/0.080",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in compact_tex]
    return Check(
        "arXiv Figure 4 matches CTA/think-budget artifacts",
        len(cta_rows) == 8 and len(budget_rows) == 16 and not mismatches and not missing_phrases,
        "mismatches="
        + ("; ".join(mismatches) if mismatches else "none")
        + "; missing_phrases="
        + (", ".join(missing_phrases) if missing_phrases else "none"),
    )


def check_arxiv_release_scope_alignment(all_tex: str) -> Check:
    docs = "\n".join(
        [
            (REPO_ROOT / "README.md").read_text(encoding="utf-8"),
            (REPO_ROOT / "docs/artifacts.md").read_text(encoding="utf-8"),
            (REPO_ROOT / "ARTIFACT_LICENSES.md").read_text(encoding="utf-8"),
            (REPO_ROOT / "configs/release_artifacts.yaml").read_text(encoding="utf-8"),
        ]
    )
    paper_mentions_future_scope = all(
        phrase in all_tex
        for phrase in [
            "permutation indices",
            "model outputs",
            "planned full code release",
        ]
    )
    docs_explain_boundary = all(
        phrase in docs
        for phrase in [
            "upstream dataset text",
            "provider caches",
            "raw 78GB internal experiment tree",
        ]
    )
    return Check(
        "repo documents boundary for arXiv planned full-release items",
        paper_mentions_future_scope and docs_explain_boundary,
        f"paper_future_scope={paper_mentions_future_scope}; docs_boundary={docs_explain_boundary}",
    )


def check_arxiv_zip(arxiv_zip: Path | None, *, offline_only: bool = False) -> list[Check]:
    if offline_only:
        return [
            Check("latest arXiv source bundle present", True, "skipped by --offline-only"),
            Check(
                "arXiv TeX quantitative sections align with release artifacts",
                True,
                "skipped by --offline-only",
            ),
        ]
    if arxiv_zip is None:
        candidates = sorted((REPO_ROOT.parent / "arxiv_latest_post_finalization").glob("*.zip"))
        arxiv_zip = candidates[-1] if candidates else None
    if arxiv_zip is None or not arxiv_zip.exists():
        return [Check("latest arXiv source bundle present", False, "zip not found")]

    required_files = {
        "sections/3-methods.tex",
        "sections/4-findings.tex",
        "sections/6-limitations.tex",
        "supplementary/sections/1-extended_dataset_details.tex",
        "supplementary/sections/2-irt_methodology.tex",
        "supplementary/sections/4-robustness_analysis.tex",
        "supplementary/sections/7-mitigation_extended.tex",
        "supplementary/sections/10-extensibility.tex",
    }
    required_markers = {
        "screened_panel_values",
        "irt_v6_screened5_modal",
        "mitigation_screened_values",
        "decoder-cleaned accuracy-swing",
    }
    with zipfile.ZipFile(arxiv_zip) as zf:
        names = set(zf.namelist())
        missing = sorted(required_files - names)
        texts = {
            name: zf.read(name).decode("utf-8", errors="replace")
            for name in required_files
            if name in names
        }
        text = "\n".join(texts.values())
    missing_markers = sorted(marker for marker in required_markers if marker not in text)
    checks = [
        Check(
            "latest arXiv source bundle present",
            not missing,
            f"path={arxiv_zip}; missing_files={missing or 'none'}",
        ),
        Check(
            "arXiv TeX quantitative sections align with release artifacts",
            not missing_markers,
            "missing_markers=" + (", ".join(missing_markers) if missing_markers else "none"),
        ),
    ]
    if missing:
        return checks
    checks.extend(
        [
            check_arxiv_table1_alignment(texts["sections/4-findings.tex"]),
            check_arxiv_dataset_alignment(
                texts["supplementary/sections/1-extended_dataset_details.tex"],
                texts["sections/3-methods.tex"],
                texts["sections/6-limitations.tex"],
            ),
            check_arxiv_odi_alignment(),
            check_arxiv_posterior_interval_alignment(),
            check_arxiv_mitigation_alignment(),
            check_arxiv_prompt_mitigation_alignment(
                texts["sections/4-findings.tex"],
                texts["supplementary/sections/7-mitigation_extended.tex"],
            ),
            check_arxiv_release_scope_alignment(text),
        ]
    )
    return checks


def check_numeric_artifacts() -> list[Check]:
    return [
        Check(f"numeric artifact: {item.name}", item.ok, item.detail)
        for item in verify_release_artifacts()
    ]


def check_screen_sanitization() -> list[Check]:
    csv_path = REPO_ROOT / "artifacts/screens/imageset_position_reference_screen.csv"
    summary_path = REPO_ROOT / "artifacts/screens/imageset_position_reference_screen_summary.json"
    forbidden_headers = {
        "question",
        "choice",
        "choices",
        "answer_choices",
        "rationale",
        "prompt",
        "text",
        "image",
        "image_path",
    }
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = {name.lower() for name in reader.fieldnames or []}

    leaked_headers = sorted(header for header in headers if header in forbidden_headers)
    long_cells = []
    for idx, row in enumerate(rows, start=2):
        for key, value in row.items():
            if key in {"regex_flags"}:
                continue
            if value and len(value) > 240:
                long_cells.append(f"{idx}:{key}")
    return [
        Check(
            "image-set screen omits upstream content columns",
            not leaked_headers,
            "leaked_headers=" + (", ".join(leaked_headers) if leaked_headers else "none"),
        ),
        Check(
            "image-set screen has expected row count",
            len(rows) == 270,
            f"rows={len(rows)} expected=270",
        ),
        Check(
            "image-set screen cells are compact identifiers/labels",
            not long_cells,
            "long_cells=" + (", ".join(long_cells[:10]) if long_cells else "none"),
        ),
        Check(
            "image-set screen summary exists",
            summary_path.exists(),
            str(summary_path.relative_to(REPO_ROOT)),
        ),
    ]


def iter_text_files() -> list[Path]:
    out = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(REPO_ROOT).parts)
        if parts & SKIP_DIRS:
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        out.append(path)
    return out


def check_secrets() -> list[Check]:
    hits = []
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                hits.append(f"{rel(path)}:{line}:{label}")
    return [
        Check(
            "text files contain no obvious credentials",
            not hits,
            "hits=" + (", ".join(hits[:20]) if hits else "none"),
        )
    ]


def check_setup_options() -> list[Check]:
    checks: list[Check] = []
    setup_path = REPO_ROOT / "setup.sh"
    env_path = REPO_ROOT / "environment.yml"
    checks.append(
        Check(
            "setup.sh exists and is executable",
            setup_path.exists() and setup_path.stat().st_mode & 0o111 != 0,
            rel(setup_path) if setup_path.exists() else "missing",
        )
    )
    if env_path.exists():
        env = yaml.safe_load(env_path.read_text(encoding="utf-8"))
        deps = [str(dep) for dep in env.get("dependencies", [])]
        env_ok = (
            env.get("name") == "facet-probe"
            and "conda-forge" in env.get("channels", [])
            and any(dep.startswith("python=3.11") for dep in deps)
            and any(dep.startswith("pip") for dep in deps)
        )
        checks.append(Check("environment.yml defines conda setup", env_ok, rel(env_path)))
    else:
        checks.append(Check("environment.yml defines conda setup", False, "missing"))

    for mode in ["conda", "uv"]:
        proc = subprocess.run(
            ["bash", "setup.sh", mode, "--dry-run"],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        checks.append(
            Check(
                f"setup.sh {mode} dry-run succeeds",
                proc.returncode == 0,
                (proc.stdout + proc.stderr).strip().splitlines()[0]
                if (proc.stdout or proc.stderr)
                else "no output",
            )
        )

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    docs_ok = "bash setup.sh conda" in readme and "bash setup.sh uv" in readme
    checks.append(
        Check(
            "README documents conda and uv setup",
            docs_ok,
            "conda+uv documented" if docs_ok else "missing setup command",
        )
    )
    return checks


def check_reproducibility_matrix() -> list[Check]:
    path = REPO_ROOT / "artifacts/reproducibility_matrix.md"
    if not path.exists():
        return [Check("reproducibility matrix exists", False, "missing")]
    text = path.read_text(encoding="utf-8")
    required_markers = [
        "Main Figure 2 and Table 1",
        "Main Table 2 ODI decomposition",
        "Appendix ODI posterior intervals",
        "Image-set screen",
        "Mixed-modality sem-flip cells",
        "decoder-noise decomposition",
        "Main Table 4",
        "Main Figure 4",
        "facet-probe verify-artifacts",
        "scripts/audit_release.py",
        "scripts/materialize_release_artifacts.py",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    return [
        Check(
            "reproducibility matrix maps paper results to artifacts",
            not missing,
            "missing=" + (", ".join(missing) if missing else "none"),
        )
    ]


def run_audit(
    source_root: Path | None,
    arxiv_zip: Path | None,
    *,
    offline_only: bool = False,
) -> list[Check]:
    manifest = load_manifest()
    checks: list[Check] = []
    checks.extend(check_manifest_coverage(manifest))
    checks.extend(check_config_artifact_coverage())
    checks.extend(
        check_materialized_reproducibility(
            manifest,
            source_root,
            offline_only=offline_only,
        )
    )
    checks.extend(check_arxiv_zip(arxiv_zip, offline_only=offline_only))
    checks.extend(check_numeric_artifacts())
    checks.extend(check_screen_sanitization())
    checks.extend(check_secrets())
    checks.extend(check_setup_options())
    checks.extend(check_reproducibility_matrix())
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional path to the EMNLP_2026 paper workspace for regeneration checks.",
    )
    parser.add_argument(
        "--arxiv-zip",
        type=Path,
        default=None,
        help="Optional path to the latest arXiv source ZIP.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Skip checks that require the private paper workspace or arXiv source ZIP.",
    )
    args = parser.parse_args()

    checks = run_audit(args.source_root, args.arxiv_zip, offline_only=args.offline_only)
    for check in checks:
        print(f"{'PASS' if check.ok else 'FAIL'}\t{check.name}\t{check.detail}")
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
