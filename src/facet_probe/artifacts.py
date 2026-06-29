"""Helpers for included release artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

ReleasePath = Path | Traversable


def repo_root() -> Path:
    """Return the source checkout root when running from a repo tree."""

    here = Path(__file__).resolve()
    candidates = [Path.cwd(), *here.parents]
    for candidate in candidates:
        if (candidate / "artifacts").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    return here.parents[2]


def release_data_root() -> ReleasePath:
    """Return release data from a checkout, or from packaged wheel resources."""

    here = Path(__file__).resolve()
    candidates = [Path.cwd(), *here.parents]
    for candidate in candidates:
        if (candidate / "artifacts").exists() and (candidate / "configs").exists():
            return candidate
    return files("facet_probe") / "release"


def artifact_path(*parts: str) -> ReleasePath:
    path: ReleasePath = release_data_root() / "artifacts"
    for part in parts:
        path = path / part
    return path


def load_json(*parts: str) -> Any:
    return json.loads(artifact_path(*parts).read_text(encoding="utf-8"))


def load_csv(*parts: str) -> list[dict[str, str]]:
    with artifact_path(*parts).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass(frozen=True)
class ArtifactCheck:
    name: str
    ok: bool
    detail: str


def verify_release_artifacts(tolerance: float = 5e-4) -> list[ArtifactCheck]:
    """Run lightweight consistency checks over shipped CSV/JSON artifacts."""

    checks: list[ArtifactCheck] = []
    rows = load_csv("paper", "per_facet_per_model.csv")
    panel = load_csv("paper", "panel_means.csv")
    mean_row = {row["facet"]: float(row["flip_rate"]) for row in panel}

    for facet in [
        "option_order",
        "evidence_chunk_order",
        "document_rank_order",
        "image_set_order_screened",
        "mixed_modality_order",
        "mean5",
    ]:
        values = [float(row[facet]) for row in rows]
        computed = sum(values) / len(values)
        expected = mean_row[facet]
        ok = abs(computed - expected) <= tolerance
        checks.append(
            ArtifactCheck(
                name=f"panel mean {facet}",
                ok=ok,
                detail=f"computed={computed:.6f} expected={expected:.6f}",
            )
        )

    odi = load_csv("odi", "facet_decomposition.csv")
    option = next(row for row in odi if row["facet"] == "option_order")
    image = next(row for row in odi if row["facet"] == "image_set_order")
    sigma_ratio = float(image["sigma_pi_median"]) / float(option["sigma_pi_median"])
    delta_field = "delta_abs_mean" if "delta_abs_mean" in image else "abs_delta_mean"
    delta_ratio = float(image[delta_field]) / float(option[delta_field])
    checks.append(
        ArtifactCheck(
            name="ODI image/option sigma ratio",
            ok=abs(sigma_ratio - 1.734) < 0.01,
            detail=f"computed={sigma_ratio:.3f} expected~=1.734",
        )
    )
    checks.append(
        ArtifactCheck(
            name="ODI image/option delta ratio",
            ok=abs(delta_ratio - 8.250) < 0.02,
            detail=f"computed={delta_ratio:.3f} expected~=8.250",
        )
    )

    cta_rows = {
        (row["dataset_id"], row["model"]): row
        for row in load_csv("mitigation", "cta_flip_summary.csv")
    }
    medx_pro = cta_rows[("medxpertqa", "gemini-3.1-pro-preview")]
    mathvision_pro = cta_rows[("mathvision", "gemini-3.1-pro-preview")]
    checks.append(
        ArtifactCheck(
            name="CTA MedXpertQA Pro reduction",
            ok=(
                abs(float(medx_pro["baseline_flip_rate"]) - 0.30) < tolerance
                and abs(float(medx_pro["cta_flip_rate"]) - 0.18) < tolerance
            ),
            detail=(
                f"baseline={float(medx_pro['baseline_flip_rate']):.3f} "
                f"cta={float(medx_pro['cta_flip_rate']):.3f}"
            ),
        )
    )
    checks.append(
        ArtifactCheck(
            name="CTA MathVision Pro null effect",
            ok=(
                abs(float(mathvision_pro["baseline_flip_rate"]) - 0.29) < tolerance
                and abs(float(mathvision_pro["cta_flip_rate"]) - 0.29) < tolerance
            ),
            detail=(
                f"baseline={float(mathvision_pro['baseline_flip_rate']):.3f} "
                f"cta={float(mathvision_pro['cta_flip_rate']):.3f}"
            ),
        )
    )

    budget_rows = {
        (row["dataset_id"], row["model"], row["budget_label"]): row
        for row in load_csv("mitigation", "think_budget_sweep.csv")
    }
    medx_pro_1k = budget_rows[("medxpertqa", "gemini-3.1-pro-preview", "1k")]
    medx_pro_24k = budget_rows[("medxpertqa", "gemini-3.1-pro-preview", "24k")]
    checks.append(
        ArtifactCheck(
            name="think-budget MedXpertQA Pro endpoint reduction",
            ok=(
                abs(float(medx_pro_1k["flip_rate"]) - 0.41) < tolerance
                and abs(float(medx_pro_24k["flip_rate"]) - 0.28) < tolerance
            ),
            detail=(
                f"1k={float(medx_pro_1k['flip_rate']):.3f} "
                f"24k={float(medx_pro_24k['flip_rate']):.3f}"
            ),
        )
    )
    return checks
