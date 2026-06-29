#!/usr/bin/env python3
"""Audit the public Facet-Probe release tree.

The audit is intentionally offline and deterministic. It checks that every file
under artifacts/ is manifest-listed, compact numeric artifacts match the public
verifier, sanitized screens do not expose upstream content, setup commands are
documented, and no obvious credentials are present in text files.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

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

ARXIV_REQUIRED_FILES = {
    "main.tex",
    "sections/1-introduction.tex",
    "sections/3-methods.tex",
    "sections/4-findings.tex",
    "sections/6-limitations.tex",
    "supplementary/sections/1-extended_dataset_details.tex",
    "supplementary/sections/2-irt_methodology.tex",
    "supplementary/sections/3-per_facet_per_model_tables.tex",
    "supplementary/sections/4-robustness_analysis.tex",
    "supplementary/sections/5-llm_judge_methodology.tex",
    "supplementary/sections/6-additional_facet_results.tex",
    "supplementary/sections/7-mitigation_extended.tex",
    "supplementary/sections/8-mixed_modality_facet.tex",
    "supplementary/sections/9-mechanism_extended.tex",
    "supplementary/sections/10-extensibility.tex",
}


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def load_manifest() -> dict[str, Any]:
    path = REPO_ROOT / "configs/release_artifacts.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_structured(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def round_half_up(value: str | float, digits: int) -> float:
    quant = Decimal("1").scaleb(-digits)
    return float(Decimal(str(value)).quantize(quant, rounding=ROUND_HALF_UP))


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda idx: values[idx])
        out = [0.0] * len(values)
        start = 0
        while start < len(values):
            end = start
            while end + 1 < len(values) and values[order[end + 1]] == values[order[start]]:
                end += 1
            rank = (start + end + 2) / 2
            for idx in range(start, end + 1):
                out[order[idx]] = rank
            start = end + 1
        return out

    if len(xs) != len(ys) or not xs:
        raise ValueError("spearman_rho requires equal-length non-empty inputs")
    rx = ranks(xs)
    ry = ranks(ys)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry, strict=True))
    denom_x = sum((x - mean_x) ** 2 for x in rx)
    denom_y = sum((y - mean_y) ** 2 for y in ry)
    return numerator / (denom_x * denom_y) ** 0.5


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


def load_arxiv_sources(
    *,
    arxiv_zip: Path | None = None,
    arxiv_source: Path | None = None,
) -> tuple[list[Check], dict[str, str]]:
    if arxiv_zip is None and arxiv_source is None:
        return [], {}
    if arxiv_zip is not None and arxiv_source is not None:
        return [
            Check(
                "arXiv source input is unambiguous",
                False,
                "pass only one of --arxiv-zip or --arxiv-source",
            )
        ], {}

    texts: dict[str, str] = {}
    if arxiv_zip is not None:
        if not arxiv_zip.exists():
            return [Check("arXiv source files present", False, f"missing {arxiv_zip}")], {}
        with zipfile.ZipFile(arxiv_zip) as zf:
            names = set(zf.namelist())
            missing = sorted(ARXIV_REQUIRED_FILES - names)
            for name in sorted(ARXIV_REQUIRED_FILES & names):
                texts[name] = zf.read(name).decode("utf-8", errors="replace")
        detail = f"path={arxiv_zip}; missing={missing or 'none'}"
    else:
        assert arxiv_source is not None
        if not arxiv_source.exists():
            return [Check("arXiv source files present", False, f"missing {arxiv_source}")], {}
        missing = []
        for name in sorted(ARXIV_REQUIRED_FILES):
            path = arxiv_source / name
            if path.exists():
                texts[name] = path.read_text(encoding="utf-8")
            else:
                missing.append(name)
        detail = f"path={arxiv_source}; missing={missing or 'none'}"

    return [Check("arXiv source files present", not missing, detail)], texts


def strip_tex_cell(text: str) -> str:
    text = text.replace("\\\\", "")
    text = re.sub(r"\\(?:textbf|underline|emph|boldsymbol)\{([^{}]*)\}", r"\1", text)
    text = text.replace("$", "").replace("~", " ")
    text = re.sub(r"\\[A-Za-z]+(?:\{\})?", "", text)
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
        if "&" not in line or "\\\\" not in line or "midrule" in line:
            continue
        parts = [strip_tex_cell(part) for part in line.split("&")]
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
        display = row["display"]
        artifact_values = [round_half_up(row[column], 2) for column in columns]
        if tex_rows.get(display) != artifact_values:
            mismatches.append(f"{display}: artifact={artifact_values} tex={tex_rows.get(display)}")
    missing = sorted({row["display"] for row in artifact_rows} - set(tex_rows))
    extra = sorted(set(tex_rows) - {row["display"] for row in artifact_rows})
    return Check(
        "arXiv Table 1 matches per-model artifact values",
        len(artifact_rows) == 18 and not mismatches and not missing and not extra,
        "mismatches="
        + ("; ".join(mismatches[:6]) if mismatches else "none")
        + f"; missing={missing or 'none'}; extra={extra or 'none'}",
    )


def check_arxiv_dataset_alignment(texts: dict[str, str]) -> Check:
    rows = load_csv_rows(REPO_ROOT / "artifacts/paper/dataset_summary.csv")
    expected_n = {
        "mmlu_pro": "200",
        "commonsenseqa": "200",
        "medxpertqa": "150",
        "mathvision": "190",
        "hotpotqa": "199",
        "musique": "200",
        "multihop_rag": "171",
        "mantis_eval": "70 raw / 18 clean",
        "medframeqa": "200 raw / 195 clean",
        "mramg": "197",
        "mmdocrag": "200",
        "mmqa": "200",
    }
    by_dataset = {row["dataset"]: row for row in rows}
    mismatches = [
        f"{name}: artifact={by_dataset.get(name, {}).get('audited_n')} tex={expected}"
        for name, expected in expected_n.items()
        if by_dataset.get(name, {}).get("audited_n") != expected
    ]
    paper_text = "\n".join(texts.values())
    required_phrases = [
        "70 (18 clean)",
        "200 (195 clean)",
        "N{=}597",
        "52/70 Mantis-Eval items excluded; 5/200",
        "18 Mantis-Eval and 195 MedFrameQA items",
    ]
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


def check_arxiv_cta_think_alignment(findings_tex: str, supplement_tex: str) -> Check:
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


def check_arxiv_robustness_alignment() -> Check:
    rows = load_csv_rows(REPO_ROOT / "artifacts/robustness/decoder_decomp_screened_by_facet.csv")
    by_key = {
        (row["model"], row["facet"], float(row["temperature"])): row
        for row in rows
    }
    mismatches = []
    expected_delta = {
        ("gemini-3-flash-preview", "option_order"): (0.049, 0.022),
        ("gemini-3.1-pro-preview", "option_order"): (0.052, 0.017),
        ("gemini-3-flash-preview", "evidence_chunk_order"): (0.048, 0.013),
        ("gemini-3.1-pro-preview", "evidence_chunk_order"): (0.051, 0.013),
        ("gemini-3-flash-preview", "document_rank_order"): (0.022, 0.008),
        ("gemini-3.1-pro-preview", "document_rank_order"): (0.024, 0.008),
    }
    for (model, facet), expected_values in expected_delta.items():
        t0 = by_key[(model, facet, 0.0)]
        t07 = by_key[(model, facet, 0.7)]
        t10 = by_key[(model, facet, 1.0)]
        values = (
            round_half_up(t0["delta_ordering"], 3),
            round_half_up((float(t07["delta_ordering"]) + float(t10["delta_ordering"])) / 2, 3),
        )
        if values != expected_values:
            mismatches.append(f"{model}/{facet}: delta={values} tex={expected_values}")

    for model, expected_values in {
        "gemini-3-flash-preview": (0.173, 0.141),
        "gemini-3.1-pro-preview": (0.134, 0.102),
    }.items():
        t0 = by_key[(model, "option_order", 0.0)]
        t07 = by_key[(model, "option_order", 0.7)]
        t10 = by_key[(model, "option_order", 1.0)]
        values = (
            round_half_up(t0["acc_swing"], 3),
            round_half_up((float(t07["acc_swing"]) + float(t10["acc_swing"])) / 2, 3),
        )
        if values != expected_values:
            mismatches.append(f"{model}/option_order: swing={values} tex={expected_values}")

    for facet, expected_values in {
        "option_order": (0.121, 0.458),
        "evidence_chunk_order": (0.100, 0.291),
        "document_rank_order": (0.052, 0.164),
    }.items():
        gemini = []
        local = []
        for row in rows:
            if row["facet"] != facet or float(row["temperature"]) == 0.0:
                continue
            target = gemini if row["model"].startswith("gemini") else local
            target.append(float(row["acc_swing"]))
        values = (
            round_half_up(sum(gemini) / len(gemini), 3),
            round_half_up(sum(local) / len(local), 3),
        )
        if values != expected_values:
            mismatches.append(f"{facet}: swing means={values} tex={expected_values}")

    return Check(
        "arXiv robustness appendix matches decoder-decomposition artifacts",
        not mismatches,
        "mismatches=" + ("; ".join(mismatches) if mismatches else "none"),
    )


def check_arxiv_mixed_modality_alignment() -> Check:
    rows = load_csv_rows(REPO_ROOT / "artifacts/paper/mixed_modality_cells.csv")
    models = yaml.safe_load((REPO_ROOT / "configs/models.yaml").read_text(encoding="utf-8"))[
        "models"
    ]
    values = [float(row["sem_flip"]) for row in rows]
    frontier = [
        float(row["sem_flip"])
        for row in rows
        if models[row["model"]]["cluster"] == "frontier"
    ]
    open_weight = [
        float(row["sem_flip"])
        for row in rows
        if models[row["model"]]["cluster"] == "open_weight"
    ]
    mismatches = []
    if (len(rows), round_half_up(min(values), 2), round_half_up(max(values), 2)) != (
        54,
        0.09,
        0.89,
    ):
        mismatches.append("mixed-modality span/count mismatch")
    for label, group, expected_median in [
        ("frontier", frontier, 0.28),
        ("open_weight", open_weight, 0.62),
    ]:
        group = sorted(group)
        median = (group[len(group) // 2 - 1] + group[len(group) // 2]) / 2
        if round_half_up(median, 2) != expected_median:
            mismatches.append(f"{label}: median={round_half_up(median, 2)}")
    return Check(
        "arXiv mixed-modality appendix matches sem-flip artifacts",
        not mismatches,
        "mismatches=" + ("; ".join(mismatches) if mismatches else "none"),
    )


def check_arxiv_capability_alignment(texts: dict[str, str]) -> Check:
    panel = load_structured(REPO_ROOT / "artifacts/paper/screened_panel_values.json")[
        "table1_rows"
    ]
    theta_rows = load_structured(REPO_ROOT / "artifacts/odi/irt_v4_correct_theta.json")[
        "models"
    ]
    summary_rows = load_csv_rows(REPO_ROOT / "artifacts/paper/capability_summary.csv")
    summary = {row["metric"]: row for row in summary_rows}
    models = load_structured(REPO_ROOT / "configs/models.yaml")["models"]

    mean5 = {row["model"]: float(row["mean5"]) for row in panel}
    theta = {row["model"]: float(row["theta_mean"]) for row in theta_rows}
    by_model = {row["model"]: row for row in panel}
    four_facet_cols = [
        "option_order",
        "evidence_chunk_order",
        "document_rank_order",
        "image_set_order_screened",
    ]

    expected = {
        "spearman_theta_correct_vs_mean5": round_half_up(
            spearman_rho(
                [theta[model] for model in mean5],
                [mean5[model] for model in mean5],
            ),
            3,
        ),
        "best_model_mean5_flip": round_half_up(min(mean5.values()), 3),
        "best_of_frontier_mean5_flip": round_half_up(
            sum(mean5[model] for model in ["gemini-3.1-pro-preview", "claude-opus-4-7", "gpt-5-5"])
            / 3,
            3,
        ),
        "best_of_open_weight_mean5_flip": round_half_up(
            sum(
                mean5[model]
                for model in [
                    "qwen3-5-27b",
                    "internvl3-5-14b",
                    "kimi-vl-a3b-instruct",
                    "medgemma-27b-it",
                ]
            )
            / 4,
            3,
        ),
    }
    for model, metric in [
        ("qwen3-5-0-8b", "qwen3_5_vl_0_8b_screened4"),
        ("qwen3-5-27b", "qwen3_5_vl_27b_screened4"),
        ("internvl3-5-4b", "internvl3_5_4b_screened4"),
        ("internvl3-5-14b", "internvl3_5_14b_screened4"),
        ("internvl3-5-38b", "internvl3_5_38b_screened4"),
    ]:
        expected[metric] = round_half_up(
            sum(float(by_model[model][facet]) for facet in four_facet_cols) / len(four_facet_cols),
            3,
        )

    mismatches = []
    missing_metrics = sorted(set(expected) - set(summary))
    if missing_metrics:
        mismatches.append("missing metrics=" + ", ".join(missing_metrics))
    for metric, value in expected.items():
        if metric not in summary:
            continue
        observed = round_half_up(summary[metric]["value"], 3)
        if observed != value:
            mismatches.append(f"{metric}: artifact={observed} computed={value}")

    best_model = min(mean5, key=mean5.get)
    if best_model != "gemini-3.1-pro-preview":
        mismatches.append(f"best model={best_model}")
    if len(theta_rows) != 18 or len(panel) != 18:
        mismatches.append(f"n theta/panel={len(theta_rows)}/{len(panel)}")
    if {models[model]["cluster"] for model in mean5} != {"frontier", "open_weight"}:
        mismatches.append("model clusters missing frontier/open_weight")

    source = "\n".join(
        [
            texts["main.tex"],
            texts["sections/4-findings.tex"],
            texts["supplementary/sections/2-irt_methodology.tex"],
        ]
    )
    source_markers = [
        "13.4\\%",
        "17\\%",
        "36\\%",
        "-0.95",
        "0.57 \\to 0.29",
        "0.37 \\to 0.32",
        "38B uptick to 0.33",
    ]
    missing_source = [marker for marker in source_markers if marker not in source]
    if missing_source:
        mismatches.append("missing source markers=" + ", ".join(missing_source))

    return Check(
        "arXiv Figure 1 capability/scaling claims match public artifacts",
        not mismatches,
        "mismatches=" + ("; ".join(mismatches) if mismatches else "none"),
    )


def check_arxiv_additional_facet_alignment(texts: dict[str, str]) -> Check:
    rows = load_csv_rows(REPO_ROOT / "artifacts/paper/additional_facet_results.csv")
    by_result = {row["result"]: row for row in rows}
    expected_ranges = {
        "multichallenge_same_order_accuracy": (0.007, 0.007),
        "synthetic_dialog_exact_match_flip": (0.085, 0.185),
        "dialog_rescore_content_flip": (0.50, 0.85),
        "gsm8k_flip": (0.0, 0.005),
        "humaneval_filtered_flip": (0.32, 0.54),
        "humaneval_diagnostic_flip": (0.20, 0.58),
        "disambiguating_queries_content_flip": (0.0, 0.0),
        "disambiguating_queries_accuracy": (0.96, 1.00),
        "glaive_k10_stress_flip_pro": (0.517, 0.517),
        "glaive_k10_stress_flip_flash": (0.550, 0.550),
        "glaive_k10_stress_accuracy": (0.77, 0.78),
    }
    mismatches = []
    missing_results = sorted(set(expected_ranges) - set(by_result))
    if missing_results:
        mismatches.append("missing results=" + ", ".join(missing_results))
    for result, (low, high) in expected_ranges.items():
        if result not in by_result:
            continue
        row = by_result[result]
        observed = (round_half_up(row["value_min"], 3), round_half_up(row["value_max"], 3))
        expected = (round_half_up(low, 3), round_half_up(high, 3))
        if observed != expected:
            mismatches.append(f"{result}: artifact={observed} tex={expected}")

    source = texts["supplementary/sections/6-additional_facet_results.tex"]
    source_markers = [
        "0.007",
        "0.085$--$0.185",
        "0.50$--$0.85",
        "0$--$0.5\\%",
        "32$--$54\\%",
        "0$ of 200",
        "1{,}200 trials",
        "0.517",
        "0.550",
    ]
    missing_source = [marker for marker in source_markers if marker not in source]
    if missing_source:
        mismatches.append("missing source markers=" + ", ".join(missing_source))

    return Check(
        "arXiv additional/demoted facet appendix matches compact artifacts",
        not mismatches,
        "mismatches=" + ("; ".join(mismatches) if mismatches else "none"),
    )


def check_arxiv_diagnostic_alignment(texts: dict[str, str]) -> Check:
    calibration_rows = load_csv_rows(
        REPO_ROOT / "artifacts/diagnostics/calibration_mechanism_summary.csv"
    )
    judge_rows = load_csv_rows(REPO_ROOT / "artifacts/diagnostics/llm_judge_validation.csv")
    calibration = {row["metric"]: row for row in calibration_rows}
    judge = {(row["judge_instrument"], row["metric"], row["subset"]): row for row in judge_rows}

    expected_calibration = {
        "mismatch_min": 8.8,
        "mismatch_max": 62.1,
        "spearman_theta_vs_mismatch": -0.57,
        "mechanism_sample_size": 225.0,
        "primary_content_rationalization_non_dialog_categorical": 58.0,
        "mantis_prescreen_primary_content_rationalization": 80.0,
        "mantis_prescreen_claude_content_rationalization": 98.0,
        "mantis_prescreen_chatgpt_content_rationalization": 55.0,
        "fleiss_kappa_6_class": 0.28,
        "fleiss_kappa_3_class": 0.30,
        "aggregate_reasoning_instability_3_class": 59.0,
        "aggregate_positional_anchoring_3_class": 18.0,
        "aggregate_other_modes_3_class": 24.0,
    }
    expected_judge = {
        ("dialog_rescore", "inter_judge_cohen_kappa", "dialog-turn semantic rescore"): 0.38,
        ("dialog_rescore", "per_stratum_ari", "dialog-turn semantic rescore"): 0.53,
        (
            "mechanism_classification",
            "fleiss_kappa_3_class",
            "reasoning-instability anchoring other",
        ): 0.30,
        ("mixed_modality_semflip", "cohen_kappa", "ChatGPT-output cells"): 0.81,
        ("mixed_modality_semflip", "cohen_kappa", "Gemini-output cells"): 0.73,
        ("mixed_modality_semflip", "cohen_kappa", "Claude-output cells"): 0.59,
        ("mixed_modality_mmqa_anchor", "n_items", "MMQA short-factoid gold subset"): 181.0,
        (
            "mixed_modality_mmqa_anchor",
            "semflip_minus_gold_anchor_min",
            "Gemini MMQA temperature sweep",
        ): 4.8,
        (
            "mixed_modality_mmqa_anchor",
            "semflip_minus_gold_anchor_max",
            "Gemini MMQA temperature sweep",
        ): 11.1,
    }

    mismatches = []
    for metric, expected in expected_calibration.items():
        if metric not in calibration:
            mismatches.append(f"missing calibration metric={metric}")
            continue
        observed = round_half_up(calibration[metric]["value"], 2)
        if observed != round_half_up(expected, 2):
            mismatches.append(f"{metric}: artifact={observed} tex={expected}")
    for key, expected in expected_judge.items():
        if key not in judge:
            mismatches.append(f"missing judge metric={key}")
            continue
        observed = round_half_up(judge[key]["value"], 2)
        if observed != round_half_up(expected, 2):
            mismatches.append(f"{key}: artifact={observed} tex={expected}")

    source = "\n".join(
        [
            texts["sections/4-findings.tex"],
            texts["supplementary/sections/4-robustness_analysis.tex"],
            texts["supplementary/sections/5-llm_judge_methodology.tex"],
            texts["supplementary/sections/8-mixed_modality_facet.tex"],
            texts["supplementary/sections/9-mechanism_extended.tex"],
        ]
    )
    source_markers = [
        "8.8--62.1",
        "-0.57",
        "N{=}50",
        "n=225 items",
        "58\\%",
        "80\\%",
        "0.28",
        "0.30",
        "59\\%",
        "0.38",
        "0.53",
        "0.81",
        "0.73",
        "0.59",
        "n = 181",
        "4.8$--$11.1",
    ]
    missing_source = [marker for marker in source_markers if marker not in source]
    if missing_source:
        mismatches.append("missing source markers=" + ", ".join(missing_source))

    return Check(
        "arXiv calibration/mechanism/judge diagnostics match compact artifacts",
        not mismatches,
        "mismatches=" + ("; ".join(mismatches) if mismatches else "none"),
    )


def check_arxiv_release_scope_alignment(all_tex: str) -> Check:
    docs = "\n".join(
        [
            (REPO_ROOT / "README.md").read_text(encoding="utf-8"),
            (REPO_ROOT / "docs/artifacts.md").read_text(encoding="utf-8"),
            (REPO_ROOT / "artifacts/arxiv_audit.md").read_text(encoding="utf-8"),
            (REPO_ROOT / "configs/release_artifacts.yaml").read_text(encoding="utf-8"),
        ]
    )
    paper_promises = [
        "permutation indices",
        "model outputs",
        "aggregation scripts",
        "full prompt",
        "template adapter",
        "per-cell flip rates",
    ]
    docs_markers = [
        "full permutation manifests",
        "normalized trial outputs",
        "aggregation scripts",
        "historical full prompt dumps",
        "judge prompt transcripts",
        "expanded adapter examples",
        "model-adapter examples",
        "per-cell diagnostic tables",
        "per-item judge labels",
        "calibration-dialog transcripts",
    ]
    all_tex_lower = all_tex.lower()
    docs_lower = docs.lower()
    missing_paper = [phrase for phrase in paper_promises if phrase.lower() not in all_tex_lower]
    missing_docs = [phrase for phrase in docs_markers if phrase.lower() not in docs_lower]
    return Check(
        "repo documents boundary for arXiv planned full-release items",
        not missing_paper and not missing_docs,
        "missing_paper="
        + (", ".join(missing_paper) if missing_paper else "none")
        + "; missing_docs="
        + (", ".join(missing_docs) if missing_docs else "none"),
    )


def check_arxiv_alignment(texts: dict[str, str]) -> list[Check]:
    if not texts:
        return []
    all_tex = "\n".join(texts.values())
    return [
        check_arxiv_capability_alignment(texts),
        check_arxiv_table1_alignment(texts["sections/4-findings.tex"]),
        check_arxiv_dataset_alignment(texts),
        check_arxiv_odi_alignment(),
        check_arxiv_posterior_interval_alignment(),
        check_arxiv_mitigation_alignment(),
        check_arxiv_cta_think_alignment(
            texts["sections/4-findings.tex"],
            texts["supplementary/sections/7-mitigation_extended.tex"],
        ),
        check_arxiv_robustness_alignment(),
        check_arxiv_mixed_modality_alignment(),
        check_arxiv_additional_facet_alignment(texts),
        check_arxiv_diagnostic_alignment(texts),
        check_arxiv_release_scope_alignment(all_tex),
    ]


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


def check_ci_workflow() -> list[Check]:
    path = REPO_ROOT / ".github/workflows/tests.yml"
    if not path.exists():
        return [Check("public CI workflow exists", False, "missing")]
    text = path.read_text(encoding="utf-8")
    try:
        workflow = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [Check("public CI workflow parses", False, str(exc))]

    matrix = (
        workflow.get("jobs", {})
        .get("unit", {})
        .get("strategy", {})
        .get("matrix", {})
        .get("python-version", [])
    )
    required_versions = {"3.10", "3.11", "3.12"}
    version_set = {str(version) for version in matrix}
    markers = [
        'pip install -e ".[dev]"',
        "bash setup.sh conda --dry-run",
        "bash setup.sh uv --dry-run",
        "python -m ruff check .",
        "python -m pytest",
        "python examples/python_library_usage.py",
        "python examples/quickstart_profile.py",
        "facet-probe paper-run",
        "facet-probe verify-artifacts",
        "python scripts/audit_release.py --offline-only",
        "python -m build",
    ]
    missing = [marker for marker in markers if marker not in text]
    return [
        Check("public CI workflow parses", isinstance(workflow, dict), rel(path)),
        Check(
            "public CI covers supported Python versions",
            required_versions <= version_set,
            "versions=" + ",".join(sorted(version_set)),
        ),
        Check(
            "public CI runs release gates",
            not missing,
            "missing=" + (", ".join(missing) if missing else "none"),
        ),
    ]


def check_packaging_metadata() -> list[Check]:
    path = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    hatch = data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {})
    wheel = hatch.get("wheel", {})
    force_include = wheel.get("force-include", {})
    sdist_include = set(hatch.get("sdist", {}).get("include", []))
    required_sdist = {
        "/src",
        "/tests",
        "/configs",
        "/artifacts",
        "/docs",
        "/examples",
        "/scripts",
        "/README.md",
        "/LICENSE",
        "/pyproject.toml",
    }
    missing_sdist = sorted(required_sdist - sdist_include)
    package_ok = (
        project.get("name") == "facet-probe"
        and project.get("version") == "0.0.1"
        and project.get("license") == "Apache-2.0"
        and project.get("scripts", {}).get("facet-probe") == "facet_probe.cli:main"
    )
    return [
        Check(
            "package metadata defines public CLI",
            package_ok,
            rel(path),
        ),
        Check(
            "wheel includes release configs and artifacts",
            force_include.get("configs") == "facet_probe/release/configs"
            and force_include.get("artifacts") == "facet_probe/release/artifacts",
            "force_include=" + str(force_include),
        ),
        Check(
            "sdist includes release source tree",
            not missing_sdist,
            "missing=" + (", ".join(missing_sdist) if missing_sdist else "none"),
        ),
    ]


def check_reproducibility_matrix() -> list[Check]:
    path = REPO_ROOT / "artifacts/reproducibility_matrix.md"
    if not path.exists():
        return [Check("reproducibility matrix exists", False, "missing")]
    text = path.read_text(encoding="utf-8")
    required_markers = [
        "Main Figure 1 capability",
        "Main Figure 2 and Table 1",
        "Main Table 2 ODI decomposition",
        "Appendix ODI posterior intervals",
        "Image-set screen",
        "Mixed-modality sem-flip cells and LLM-judge validation",
        "Main Q6 confidence calibration",
        "Appendix additional facets",
        "decoder-noise decomposition",
        "Main Table 4",
        "Main Figure 4",
        "arxiv_audit.md",
        "facet-probe verify-artifacts",
        "scripts/audit_release.py --offline-only",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    return [
        Check(
            "reproducibility matrix maps paper results to artifacts",
            not missing,
            "missing=" + (", ".join(missing) if missing else "none"),
        )
    ]


def check_arxiv_audit_report() -> list[Check]:
    path = REPO_ROOT / "artifacts/arxiv_audit.md"
    if not path.exists():
        return [Check("arXiv source audit report exists", False, "missing")]
    text = path.read_text(encoding="utf-8")
    required_markers = [
        "2a432c522089dc7d899555b001461479f050c96600779a608236bd6b4061b357",
        "python scripts/audit_release.py --arxiv-zip",
        "Main Figure 1 capability",
        "Main Figure 2 / Table 1",
        "Main Table 2 ODI",
        "Main Q6 calibration",
        "Appendix demoted facets",
        "Main Table 4",
        "Main Figure 4",
        "Deferred Full-Release Items",
        "full permutation manifests",
        "normalized trial outputs",
        "historical full prompt dumps",
        "judge prompt transcripts",
        "expanded dataset-adapter templates",
        "model-adapter examples",
        "per-item judge labels",
    ]
    missing = [marker for marker in required_markers if marker not in text]
    return [
        Check(
            "arXiv source audit report documents checked and deferred items",
            not missing,
            "missing=" + (", ".join(missing) if missing else "none"),
        )
    ]


def check_public_facing_language() -> list[Check]:
    hits = []
    local_area = "work" + "space"
    patterns = [
        "For " + "authors" + " or " + "review" + "ers",
        "private paper " + local_area,
        "local paper " + local_area,
        "paper " + local_area,
        "full " + "local `EMNLP" + "_2026`",
        "arxiv" + "_latest" + "_post_finalization",
        "--source" + "-root",
        "/home/" + "user/",
    ]
    for path in iter_text_files():
        if path.name == "audit_release.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in patterns:
            if pattern in text:
                hits.append(f"{rel(path)}:{pattern}")
    return [
        Check(
            "public text avoids non-public local-context instructions",
            not hits,
            "hits=" + (", ".join(hits[:20]) if hits else "none"),
        )
    ]


def run_audit(
    *,
    arxiv_zip: Path | None = None,
    arxiv_source: Path | None = None,
) -> list[Check]:
    manifest = load_manifest()
    checks: list[Check] = []
    checks.extend(check_manifest_coverage(manifest))
    checks.extend(check_config_artifact_coverage())
    checks.extend(check_numeric_artifacts())
    checks.extend(check_screen_sanitization())
    checks.extend(check_secrets())
    checks.extend(check_setup_options())
    checks.extend(check_ci_workflow())
    checks.extend(check_packaging_metadata())
    checks.extend(check_reproducibility_matrix())
    checks.extend(check_arxiv_audit_report())
    checks.extend(check_public_facing_language())
    source_checks, arxiv_texts = load_arxiv_sources(arxiv_zip=arxiv_zip, arxiv_source=arxiv_source)
    checks.extend(source_checks)
    checks.extend(check_arxiv_alignment(arxiv_texts))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Accepted for compatibility; the public release audit is always offline.",
    )
    parser.add_argument(
        "--arxiv-zip",
        type=Path,
        default=None,
        help="Optional arXiv source ZIP for source-grounded paper-value checks.",
    )
    parser.add_argument(
        "--arxiv-source",
        type=Path,
        default=None,
        help="Optional extracted arXiv source directory for source-grounded checks.",
    )
    args = parser.parse_args()

    checks = run_audit(arxiv_zip=args.arxiv_zip, arxiv_source=args.arxiv_source)
    for check in checks:
        print(f"{'PASS' if check.ok else 'FAIL'}\t{check.name}\t{check.detail}")
    return 0 if all(check.ok for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
