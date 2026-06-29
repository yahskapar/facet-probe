"""IRT/ODI helpers for released artifacts and run exports."""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from facet_probe.artifacts import artifact_path, load_csv, load_json
from facet_probe.figures import write_irt_outcome_figures, write_released_irt_figures
from facet_probe.metrics import write_csv, write_json
from facet_probe.schema import TrialRecord

IRT_OUTCOMES = ("modal", "correct")

_ODI_ARTIFACTS = (
    "facet_decomposition.csv",
    "posterior_intervals.csv",
    "irt_v4_modal_per_facet_summary.json",
    "irt_v4_correct_per_facet_summary.json",
    "irt_v4_correct_theta.json",
    "irt_v4_modal_theta.json",
    "irt_v4_modal_diagnostics.json",
    "irt_v4_correct_diagnostics.json",
    "irt_input_summary.json",
    "screen_filter_metadata.json",
    "irt_v4_modal_per_item_params.parquet",
    "irt_v4_correct_per_item_params.parquet",
)


def released_irt_summary() -> dict[str, Any]:
    """Load compact ODI/IRT artifacts shipped with the public release."""

    modal_facet = load_json("odi", "irt_v4_modal_per_facet_summary.json")
    correct_facet = load_json("odi", "irt_v4_correct_per_facet_summary.json")
    modal_theta = load_json("odi", "irt_v4_modal_theta.json")
    correct_theta = load_json("odi", "irt_v4_correct_theta.json")
    modal_diag = load_json("odi", "irt_v4_modal_diagnostics.json")
    correct_diag = load_json("odi", "irt_v4_correct_diagnostics.json")
    input_summary = load_json("odi", "irt_input_summary.json")

    return {
        "schema_version": 1,
        "release": "v0.0.1",
        "model_family": "multi-facet Bayesian 2PL ODI/IRT",
        "paper_usage": {
            "main_table_2": "modal outcome facet decomposition",
            "capability_correlations": "correct outcome model theta summaries",
            "appendix": "posterior intervals and compact per-item posterior summaries",
        },
        "fit_note": (
            "The public release ships compact posterior summaries from the paper fit. "
            "The full posterior traces are intentionally excluded, and future releases "
            "may further optimize the fitting workflow."
        ),
        "input_summary": input_summary,
        "outcomes": {
            "modal": {
                "description": (
                    "1 when a trial answer matches that model/item's untied modal "
                    "answer across orderings; used for ordering-instability ODI."
                ),
                "per_facet_summary": modal_facet,
                "theta": modal_theta,
                "diagnostics": modal_diag,
            },
            "correct": {
                "description": (
                    "1 when the normalized model answer matches the normalized gold "
                    "answer; used for model ability summaries."
                ),
                "per_facet_summary": correct_facet,
                "theta": correct_theta,
                "diagnostics": correct_diag,
            },
        },
        "facet_decomposition": load_csv("odi", "facet_decomposition.csv"),
        "posterior_intervals": load_csv("odi", "posterior_intervals.csv"),
        "screen_filter_metadata": load_json("odi", "screen_filter_metadata.json"),
        "artifact_files": [f"artifacts/odi/{name}" for name in _ODI_ARTIFACTS],
        "analysis_commands": {
            "write_released_summary": (
                "facet-probe irt-summary --output-dir reports/released_irt"
            ),
            "export_run_trials": (
                "facet-probe irt-export runs/qwen3-5-4b-paper/trials.jsonl "
                "--output-dir runs/qwen3-5-4b-paper/irt_input"
            ),
            "fit_run_trials": (
                "facet-probe irt-fit runs/qwen3-5-4b-paper/trials.jsonl "
                "--outcome modal --output-dir runs/qwen3-5-4b-paper/irt_fit_modal"
            ),
        },
    }


def write_released_irt_summary(output_dir: str | Path) -> dict[str, Any]:
    """Write a portable summary bundle for the released ODI/IRT artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary = released_irt_summary()
    files: dict[str, str] = {}

    summary_path = output / "released_irt_summary.json"
    write_json(summary_path, summary)
    files["summary_json"] = str(summary_path)

    facet_path = output / "facet_decomposition.csv"
    posterior_path = output / "posterior_intervals.csv"
    write_csv(facet_path, summary["facet_decomposition"])
    write_csv(posterior_path, summary["posterior_intervals"])
    files["facet_decomposition_csv"] = str(facet_path)
    files["posterior_intervals_csv"] = str(posterior_path)

    diagnostics_path = output / "diagnostics.csv"
    diagnostics_rows = [
        {"outcome": outcome, **payload["diagnostics"]}
        for outcome, payload in summary["outcomes"].items()
    ]
    write_csv(diagnostics_path, diagnostics_rows)
    files["diagnostics_csv"] = str(diagnostics_path)

    for outcome in IRT_OUTCOMES:
        theta_path = output / f"theta_{outcome}.csv"
        write_csv(theta_path, summary["outcomes"][outcome]["theta"]["models"])
        files[f"theta_{outcome}_csv"] = str(theta_path)

    copied = []
    artifact_dir = output / "artifacts" / "odi"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in _ODI_ARTIFACTS:
        src = artifact_path("odi", name)
        dest = artifact_dir / name
        dest.write_bytes(src.read_bytes())
        copied.append(
            {
                "artifact": f"artifacts/odi/{name}",
                "copied_to": str(dest),
            }
        )
    manifest_path = output / "released_irt_manifest.json"
    write_json(manifest_path, {"schema_version": 1, "copied_artifacts": copied})
    files["manifest_json"] = str(manifest_path)
    files["copied_artifact_dir"] = str(artifact_dir)
    figure_paths = write_released_irt_figures(summary, output / "figures")
    files.update({f"figures_{name}": str(path) for name, path in figure_paths.items()})

    return {
        "status": "completed",
        "summary": _compact_released_status(summary),
        "files": files,
    }


def trial_records_to_irt_rows(
    records: Iterable[Mapping[str, Any] | TrialRecord],
    *,
    outcomes: Sequence[str] = IRT_OUTCOMES,
) -> list[dict[str, Any]]:
    """Convert trial JSONL rows into long-form modal/correct IRT outcome rows."""

    rows, _summary = _build_irt_export(records, outcomes=outcomes)
    return rows


def write_irt_input(
    records: Iterable[Mapping[str, Any] | TrialRecord],
    output_dir: str | Path,
    *,
    outcomes: Sequence[str] = IRT_OUTCOMES,
) -> dict[str, Any]:
    """Write IRT-compatible outcome rows and summaries for a completed run."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, summary = _build_irt_export(records, outcomes=outcomes)

    csv_path = output / "irt_input_trials.csv"
    jsonl_path = output / "irt_input_trials.jsonl"
    groups_path = output / "irt_input_groups.csv"
    summary_path = output / "irt_input_summary.json"

    write_csv(csv_path, rows)
    _write_jsonl(jsonl_path, rows)
    group_rows = _group_summary_rows(rows)
    write_csv(groups_path, group_rows)
    write_json(summary_path, summary)

    return {
        "status": "completed",
        "summary": summary,
        "files": {
            "trials_csv": str(csv_path),
            "trials_jsonl": str(jsonl_path),
            "groups_csv": str(groups_path),
            "summary_json": str(summary_path),
        },
    }


def write_irt_fit(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    outcome: str = "modal",
    n_chains: int = 4,
    n_draws: int = 1500,
    n_tune: int = 1500,
    target_accept: float = 0.95,
    nuts_sampler: str = "numpyro",
    chain_method: str = "parallel",
    seed: int = 42,
    limit_items_per_facet: int | None = None,
    dry_run: bool = False,
    save_idata: bool = False,
    progressbar: bool = False,
) -> dict[str, Any]:
    """Fit the public multi-facet Bayesian ODI/IRT model.

    ``input_path`` may be an exported ``irt_input_trials`` CSV/JSONL file or a
    raw ``facet-probe paper-run`` ``trials.jsonl`` file. Raw trials are exported
    into ``output_dir / "irt_input"`` before fitting so the deterministic
    intermediate remains inspectable.
    """

    outcomes = IRT_OUTCOMES if outcome == "both" else (outcome,)
    unknown = sorted(set(outcomes) - set(IRT_OUTCOMES))
    if unknown:
        raise ValueError(f"unknown IRT outcome(s): {', '.join(unknown)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_rows, source_info, export_status = _load_or_export_fit_source(
        input_path,
        output,
    )

    fits = []
    files: dict[str, str] = {}
    if export_status is not None:
        files.update(
            {
                f"input_export_{name}": path
                for name, path in export_status["files"].items()
            }
        )
    for label in outcomes:
        rows = _prepare_fit_rows(
            source_rows,
            outcome=label,
            limit_items_per_facet=limit_items_per_facet,
        )
        fit_input = _encode_fit_rows(rows, require_numpy=not dry_run)
        input_summary = _fit_input_summary(rows, label=label)
        input_summary_path = output / f"irt_fit_{label}_input_summary.json"
        write_json(input_summary_path, input_summary)
        files[f"{label}_input_summary_json"] = str(input_summary_path)

        if dry_run:
            fits.append(
                {
                    "outcome": label,
                    "status": "prepared",
                    "input_summary": input_summary,
                }
            )
            continue

        stack = _load_pymc_stack()
        fit_status = _fit_one_outcome(
            fit_input,
            label=label,
            output_dir=output,
            n_chains=n_chains,
            n_draws=n_draws,
            n_tune=n_tune,
            target_accept=target_accept,
            nuts_sampler=nuts_sampler,
            chain_method=chain_method,
            seed=seed,
            save_idata=save_idata,
            progressbar=progressbar,
            stack=stack,
        )
        files.update({f"{label}_{name}": path for name, path in fit_status["files"].items()})
        fits.append(fit_status)

    status = {
        "status": "prepared" if dry_run else "completed",
        "model_family": "multi-facet Bayesian 2PL ODI/IRT",
        "source": str(input_path),
        "source_type": source_info["source_type"],
        "input_export": export_status,
        "fit_note": (
            "This command fits the public paper-style model over modal/correct "
            "outcome rows. If raw trial JSONL was supplied, the deterministic "
            "IRT input export is written under the fit output directory. The "
            "implementation is intended for reproducible public use and may be "
            "further optimized in future releases."
        ),
        "settings": {
            "outcome": outcome,
            "n_chains": n_chains,
            "n_draws": n_draws,
            "n_tune": n_tune,
            "target_accept": target_accept,
            "nuts_sampler": nuts_sampler,
            "chain_method": chain_method,
            "seed": seed,
            "limit_items_per_facet": limit_items_per_facet,
            "dry_run": dry_run,
            "save_idata": save_idata,
        },
        "fits": fits,
        "files": files,
    }
    status_path = output / "irt_fit_summary.json"
    write_json(status_path, status)
    status["files"]["fit_summary_json"] = str(status_path)
    return status


def load_irt_input_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load `irt-export` CSV or JSONL rows."""

    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix.lower() == ".csv":
        import csv

        with open(path, newline="", encoding="utf-8") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError(f"unsupported IRT input extension for {path}; expected .csv or .jsonl")


def _load_or_export_fit_source(
    input_path: str | Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any] | None]:
    path = Path(input_path)
    rows = load_irt_input_rows(path)
    if _looks_like_irt_input_rows(rows):
        return rows, {"source_type": "irt_input"}, None
    if _has_partial_irt_input_schema(rows):
        raise ValueError(
            f"{path} mixes IRT input and trial schemas; expected either raw "
            "trial JSONL or an irt-export CSV/JSONL."
        )
    if path.suffix.lower() != ".jsonl":
        raise ValueError(
            f"{path} does not look like exported IRT rows. Raw run trials must "
            "be provided as JSONL, or run `facet-probe irt-export` first."
        )

    export_status = write_irt_input(rows, output_dir / "irt_input", outcomes=IRT_OUTCOMES)
    exported_rows = load_irt_input_rows(export_status["files"]["trials_csv"])
    return exported_rows, {"source_type": "trial_jsonl"}, export_status


def _looks_like_irt_input_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        "outcome" in row and "outcome_value" in row
        for row in rows
    )


def _has_partial_irt_input_schema(rows: Sequence[Mapping[str, Any]]) -> bool:
    markers = [
        ("outcome" in row, "outcome_value" in row)
        for row in rows
    ]
    return any(outcome or value for outcome, value in markers) and not all(
        outcome and value for outcome, value in markers
    )


def _build_irt_export(
    records: Iterable[Mapping[str, Any] | TrialRecord],
    *,
    outcomes: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = tuple(outcomes)
    unknown = sorted(set(requested) - set(IRT_OUTCOMES))
    if unknown:
        raise ValueError(f"unknown IRT outcome(s): {', '.join(unknown)}")

    trials = [_source_trial(record) for record in records]
    grouped: dict[tuple[str, str, str, str], list[_SourceTrial]] = defaultdict(list)
    for trial in trials:
        grouped[trial.key].append(trial)

    modal_by_key, tied_modal_keys = _modal_answers(grouped)
    rows: list[dict[str, Any]] = []
    outcome_summary: dict[str, dict[str, int]] = {}
    if "modal" in requested:
        outcome_summary["modal"] = {
            "n_rows": 0,
            "n_item_groups": 0,
            "skipped_missing_answer": 0,
            "skipped_tied_modal_trials": 0,
        }
    if "correct" in requested:
        outcome_summary["correct"] = {
            "n_rows": 0,
            "n_item_groups": 0,
            "skipped_missing_correct": 0,
        }
    outcome_item_keys: dict[str, set[tuple[str, str, str, str]]] = {
        outcome: set() for outcome in requested
    }

    for trial in sorted(trials, key=_source_trial_sort_key):
        if "modal" in requested:
            modal_answer = modal_by_key.get(trial.key)
            if trial.key in tied_modal_keys:
                outcome_summary["modal"]["skipped_tied_modal_trials"] += 1
            elif trial.answer_normalized is None:
                outcome_summary["modal"]["skipped_missing_answer"] += 1
            elif modal_answer is None:
                outcome_summary["modal"]["skipped_missing_answer"] += 1
            else:
                rows.append(
                    _irt_row(
                        trial,
                        outcome="modal",
                        value=trial.answer_normalized == modal_answer,
                        modal_answer=modal_answer,
                        n_orderings=len(grouped[trial.key]),
                    )
                )
                outcome_summary["modal"]["n_rows"] += 1
                outcome_item_keys["modal"].add(trial.key)

        if "correct" in requested:
            if trial.correct is None:
                outcome_summary["correct"]["skipped_missing_correct"] += 1
            else:
                rows.append(
                    _irt_row(
                        trial,
                        outcome="correct",
                        value=trial.correct,
                        modal_answer=modal_by_key.get(trial.key),
                        n_orderings=len(grouped[trial.key]),
                    )
                )
                outcome_summary["correct"]["n_rows"] += 1
                outcome_item_keys["correct"].add(trial.key)

    for outcome, keys in outcome_item_keys.items():
        outcome_summary[outcome]["n_item_groups"] = len(keys)

    summary = {
        "schema_version": 1,
        "description": (
            "Long-form Bernoulli outcome rows for ODI/IRT-style analysis. "
            "This export prepares inputs; it does not perform posterior fitting."
        ),
        "outcomes": outcome_summary,
        "n_source_trials": len(trials),
        "n_source_item_groups": len(grouped),
        "n_rows": len(rows),
        "n_models": len({trial.model for trial in trials}),
        "facets": sorted({trial.facet for trial in trials}),
        "datasets": sorted({trial.dataset for trial in trials}),
        "models": sorted({trial.model for trial in trials}),
        "fit_note": (
            "The paper fit used a multi-facet Bayesian 2PL model over these "
            "modal/correct outcomes. Future releases may further optimize the "
            "posterior fitting workflow."
        ),
    }
    return rows, summary


def _prepare_fit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    outcome: str,
    limit_items_per_facet: int | None,
) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        if str(row.get("outcome") or "") != outcome:
            continue
        value = _coerce_binary(row.get("outcome_value"))
        if value is None:
            continue
        facet = str(row.get("facet") or "")
        dataset = str(row.get("dataset") or "")
        model = str(row.get("model") or "")
        item_id = str(row.get("item_id") or "")
        if not facet or not dataset or not model or not item_id:
            continue
        ordering_idx = int(row.get("ordering_idx") or 0)
        item_key = str(row.get("item_key") or "::".join([facet, dataset, item_id]))
        prepared.append(
            {
                "outcome": outcome,
                "outcome_value": value,
                "facet": facet,
                "dataset": dataset,
                "model": model,
                "item_id": item_id,
                "item_key": item_key,
                "fd_key": "::".join([facet, dataset]),
                "fdo_key": "::".join([facet, dataset, str(ordering_idx)]),
                "ordering_idx": ordering_idx,
            }
        )

    if limit_items_per_facet is not None:
        prepared = _limit_fit_items(prepared, limit_items_per_facet)

    if not prepared:
        raise ValueError(f"no usable IRT rows found for outcome={outcome!r}")
    return sorted(
        prepared,
        key=lambda row: (
            str(row["facet"]),
            str(row["dataset"]),
            str(row["model"]),
            str(row["item_id"]),
            int(row["ordering_idx"]),
        ),
    )


def _limit_fit_items(rows: Sequence[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("--limit-items-per-facet must be positive")
    keys_by_facet: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row["facet"]), str(row["item_key"]))
        if key in seen:
            continue
        seen.add(key)
        keys_by_facet[key[0]].append(key[1])
    keep = {
        (facet, item_key)
        for facet, item_keys in keys_by_facet.items()
        for item_key in sorted(item_keys)[:limit]
    }
    return [row for row in rows if (str(row["facet"]), str(row["item_key"])) in keep]


def _encode_fit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    require_numpy: bool,
) -> dict[str, Any]:
    items = sorted({str(row["item_key"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows})
    facets = sorted({str(row["facet"]) for row in rows})
    fds = sorted({str(row["fd_key"]) for row in rows})
    fdos = sorted({str(row["fdo_key"]) for row in rows})
    item_to_idx = {key: idx for idx, key in enumerate(items)}
    model_to_idx = {key: idx for idx, key in enumerate(models)}
    facet_to_idx = {key: idx for idx, key in enumerate(facets)}
    fd_to_idx = {key: idx for idx, key in enumerate(fds)}
    fdo_to_idx = {key: idx for idx, key in enumerate(fdos)}

    item_facet = {}
    item_fd = {}
    fdo_facet = {}
    for row in rows:
        item_facet.setdefault(str(row["item_key"]), str(row["facet"]))
        item_fd.setdefault(str(row["item_key"]), str(row["fd_key"]))
        fdo_facet.setdefault(str(row["fdo_key"]), str(row["facet"]))

    item_idx_list = [item_to_idx[str(row["item_key"])] for row in rows]
    model_idx_list = [model_to_idx[str(row["model"])] for row in rows]
    fdo_idx_list = [fdo_to_idx[str(row["fdo_key"])] for row in rows]
    item_facet_idx_list = [facet_to_idx[item_facet[key]] for key in items]
    item_fd_idx_list = [fd_to_idx[item_fd[key]] for key in items]
    fdo_facet_idx_list = [facet_to_idx[fdo_facet[key]] for key in fdos]
    y_list = [int(row["outcome_value"]) for row in rows]

    pairs = sorted(set(zip(item_idx_list, fdo_idx_list, strict=True)))
    pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
    trial_pair_idx_list = [
        pair_to_idx[(item_idx, fdo_idx)]
        for item_idx, fdo_idx in zip(item_idx_list, fdo_idx_list, strict=True)
    ]

    encoded: dict[str, Any] = {
        "rows": list(rows),
        "items": items,
        "models": models,
        "facets": facets,
        "fds": fds,
        "fdos": fdos,
        "item_facet_by_key": item_facet,
        "item_fd_by_key": item_fd,
        "fdo_facet_by_key": fdo_facet,
    }
    if not require_numpy:
        encoded.update(
            {
                "item_idx": item_idx_list,
                "model_idx": model_idx_list,
                "fdo_idx": fdo_idx_list,
                "item_facet_idx": item_facet_idx_list,
                "item_fd_idx": item_fd_idx_list,
                "fdo_facet_idx": fdo_facet_idx_list,
                "pairs": pairs,
                "trial_pair_idx": trial_pair_idx_list,
                "y": y_list,
            }
        )
        return encoded

    np = _load_numpy()
    encoded.update(
        {
            "item_idx": np.array(item_idx_list, dtype=np.int32),
            "model_idx": np.array(model_idx_list, dtype=np.int32),
            "fdo_idx": np.array(fdo_idx_list, dtype=np.int32),
            "item_facet_idx": np.array(item_facet_idx_list, dtype=np.int32),
            "item_fd_idx": np.array(item_fd_idx_list, dtype=np.int32),
            "fdo_facet_idx": np.array(fdo_facet_idx_list, dtype=np.int32),
            "pair_item_idx": np.array([pair[0] for pair in pairs], dtype=np.int32),
            "trial_pair_idx": np.array(trial_pair_idx_list, dtype=np.int32),
            "y": np.array(y_list, dtype=np.int32),
        }
    )
    return encoded


def _fit_one_outcome(
    fit_input: Mapping[str, Any],
    *,
    label: str,
    output_dir: Path,
    n_chains: int,
    n_draws: int,
    n_tune: int,
    target_accept: float,
    nuts_sampler: str,
    chain_method: str,
    seed: int,
    save_idata: bool,
    progressbar: bool,
    stack: Mapping[str, Any],
) -> dict[str, Any]:
    idata = _sample_pymc_model(
        fit_input,
        n_chains=n_chains,
        n_draws=n_draws,
        n_tune=n_tune,
        target_accept=target_accept,
        nuts_sampler=nuts_sampler,
        chain_method=chain_method,
        seed=seed,
        progressbar=progressbar,
        stack=stack,
    )
    item_rows = _extract_item_params(idata, fit_input)
    theta_rows = _extract_theta(idata, fit_input)
    diagnostics = _diagnostics(idata, stack)
    per_facet = _per_facet_summary(idata, fit_input, item_rows)
    facet_rows = _facet_summary_rows(per_facet)

    item_path = output_dir / f"irt_v4_{label}_per_item_params.csv"
    facet_path = output_dir / f"irt_v4_{label}_per_facet_summary.json"
    facet_csv_path = output_dir / f"irt_v4_{label}_facet_decomposition.csv"
    theta_path = output_dir / f"irt_v4_{label}_theta.json"
    theta_csv_path = output_dir / f"irt_v4_{label}_theta.csv"
    diagnostics_path = output_dir / f"irt_v4_{label}_diagnostics.json"
    write_csv(item_path, item_rows)
    write_json(facet_path, per_facet)
    write_csv(facet_csv_path, facet_rows)
    write_json(theta_path, {"models": theta_rows})
    write_csv(theta_csv_path, theta_rows)
    write_json(diagnostics_path, diagnostics)
    files = {
        "per_item_csv": str(item_path),
        "per_facet_summary_json": str(facet_path),
        "facet_decomposition_csv": str(facet_csv_path),
        "theta_json": str(theta_path),
        "theta_csv": str(theta_csv_path),
        "diagnostics_json": str(diagnostics_path),
    }
    figure_paths = write_irt_outcome_figures(
        output_dir / "figures",
        outcome=label,
        theta_rows=theta_rows,
        facet_rows=facet_rows,
    )
    files.update({f"figures_{name}": str(path) for name, path in figure_paths.items()})
    if save_idata:
        idata_path = output_dir / f"irt_v4_{label}_idata.nc"
        idata.to_netcdf(idata_path)
        files["idata_nc"] = str(idata_path)
    return {
        "outcome": label,
        "status": "completed",
        "diagnostics": diagnostics,
        "input_summary": _fit_input_summary(fit_input["rows"], label=label),
        "files": files,
    }


def _sample_pymc_model(
    fit_input: Mapping[str, Any],
    *,
    n_chains: int,
    n_draws: int,
    n_tune: int,
    target_accept: float,
    nuts_sampler: str,
    chain_method: str,
    seed: int,
    progressbar: bool,
    stack: Mapping[str, Any],
) -> Any:
    pm = stack["pm"]
    pt = stack["pt"]
    coords = {
        "model": fit_input["models"],
        "item": fit_input["items"],
        "facet": fit_input["facets"],
        "fd": fit_input["fds"],
        "fdo": fit_input["fdos"],
        "pair": list(range(len(fit_input["pair_item_idx"]))),
    }
    y = fit_input["y"]

    with pm.Model(coords=coords):
        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=1.0, dims="fd")
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5, dims="fd")
        beta_raw = pm.Normal("beta_raw", mu=0.0, sigma=1.0, dims="item")
        beta = pm.Deterministic(
            "beta",
            mu_beta[fit_input["item_fd_idx"]]
            + sigma_beta[fit_input["item_fd_idx"]] * beta_raw,
            dims="item",
        )

        alpha = pm.LogNormal("alpha", mu=0.0, sigma=0.3, dims="item")

        sigma_delta_facet = pm.HalfNormal("sigma_delta", sigma=0.3, dims="facet")
        delta_raw = pm.Normal("delta_raw", mu=0.0, sigma=1.0, dims="fdo")
        delta = pm.Deterministic(
            "delta",
            sigma_delta_facet[fit_input["fdo_facet_idx"]] * delta_raw,
            dims="fdo",
        )

        mu_log_sigma = pm.Normal("mu_log_sigma", mu=-1.2, sigma=0.5, dims="facet")
        tau_log_sigma = pm.HalfNormal("tau_log_sigma", sigma=0.1, dims="facet")
        log_sigma_raw = pm.Normal("log_sigma_raw", mu=0.0, sigma=1.0, dims="item")
        log_sigma_item = pm.Deterministic(
            "log_sigma_item",
            mu_log_sigma[fit_input["item_facet_idx"]]
            + tau_log_sigma[fit_input["item_facet_idx"]] * log_sigma_raw,
            dims="item",
        )
        sigma_item = pm.Deterministic("sigma_item", pm.math.exp(log_sigma_item), dims="item")

        z = pm.Normal("z", mu=0.0, sigma=1.0, dims="pair")
        gamma = pm.Deterministic("gamma", sigma_item[fit_input["pair_item_idx"]] * z, dims="pair")

        theta_raw = pm.Normal("theta_raw", mu=0.0, sigma=1.0, dims="model")
        theta = pm.Deterministic("theta", theta_raw - pt.mean(theta_raw), dims="model")

        logit_p = alpha[fit_input["item_idx"]] * (
            theta[fit_input["model_idx"]]
            - beta[fit_input["item_idx"]]
            - delta[fit_input["fdo_idx"]]
            - gamma[fit_input["trial_pair_idx"]]
        )
        pm.Bernoulli("y", logit_p=logit_p, observed=y)

        kwargs: dict[str, Any] = {
            "draws": n_draws,
            "tune": n_tune,
            "chains": n_chains,
            "target_accept": target_accept,
            "random_seed": seed,
            "return_inferencedata": True,
            "progressbar": progressbar,
        }
        if nuts_sampler != "pymc":
            kwargs["nuts_sampler"] = nuts_sampler
            if chain_method:
                kwargs["chain_method"] = chain_method
        return pm.sample(**kwargs)


def _extract_item_params(idata: Any, fit_input: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    means = {
        name: _posterior_mean(idata, name)
        for name in ("beta", "alpha", "sigma_item")
    }
    lows = {
        name: _posterior_quantile(idata, name, 0.025)
        for name in ("beta", "alpha", "sigma_item")
    }
    highs = {
        name: _posterior_quantile(idata, name, 0.975)
        for name in ("beta", "alpha", "sigma_item")
    }
    by_item = _first_row_by_item(fit_input["rows"])
    for idx, item_key in enumerate(fit_input["items"]):
        source = by_item[item_key]
        rows.append(
            {
                "item_key": item_key,
                "facet": source["facet"],
                "dataset": source["dataset"],
                "item_id": source["item_id"],
                "beta": _float(means["beta"][idx]),
                "beta_2.5": _float(lows["beta"][idx]),
                "beta_97.5": _float(highs["beta"][idx]),
                "alpha": _float(means["alpha"][idx]),
                "alpha_2.5": _float(lows["alpha"][idx]),
                "alpha_97.5": _float(highs["alpha"][idx]),
                "sigma_pi": _float(means["sigma_item"][idx]),
                "sigma_pi_2.5": _float(lows["sigma_item"][idx]),
                "sigma_pi_97.5": _float(highs["sigma_item"][idx]),
            }
        )
    return rows


def _extract_theta(idata: Any, fit_input: Mapping[str, Any]) -> list[dict[str, Any]]:
    mean = _posterior_mean(idata, "theta")
    low = _posterior_quantile(idata, "theta", 0.025)
    high = _posterior_quantile(idata, "theta", 0.975)
    return [
        {
            "model": model,
            "theta_mean": _float(mean[idx]),
            "theta_2.5": _float(low[idx]),
            "theta_97.5": _float(high[idx]),
        }
        for idx, model in enumerate(fit_input["models"])
    ]


def _per_facet_summary(
    idata: Any,
    fit_input: Mapping[str, Any],
    item_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    np = _load_numpy()
    delta_mean = _posterior_mean(idata, "delta")
    fdo_facet = fit_input["fdo_facet_by_key"]
    out = {"facets": {}}
    for facet in fit_input["facets"]:
        sub = [row for row in item_rows if row["facet"] == facet]
        sigmas = np.array([float(row["sigma_pi"]) for row in sub], dtype=float)
        fdo_indices = [
            idx
            for idx, key in enumerate(fit_input["fdos"])
            if fdo_facet[str(key)] == facet
        ]
        deltas = np.array([float(delta_mean[idx]) for idx in fdo_indices], dtype=float)
        out["facets"][facet] = {
            "n_items": int(len(sub)),
            "sigma_pi_mean": _float(sigmas.mean()) if len(sigmas) else None,
            "sigma_pi_median": _float(np.median(sigmas)) if len(sigmas) else None,
            "sigma_pi_q25": _float(np.quantile(sigmas, 0.25)) if len(sigmas) else None,
            "sigma_pi_q75": _float(np.quantile(sigmas, 0.75)) if len(sigmas) else None,
            "sigma_pi_min": _float(sigmas.min()) if len(sigmas) else None,
            "sigma_pi_max": _float(sigmas.max()) if len(sigmas) else None,
            "delta_n": int(len(deltas)),
            "delta_mean": _float(deltas.mean()) if len(deltas) else None,
            "delta_abs_mean": _float(np.abs(deltas).mean()) if len(deltas) else None,
            "delta_min": _float(deltas.min()) if len(deltas) else None,
            "delta_max": _float(deltas.max()) if len(deltas) else None,
        }
    option = out["facets"].get("option_order")
    if option and option.get("sigma_pi_median") and option.get("delta_abs_mean"):
        for facet, row in out["facets"].items():
            if facet == "option_order":
                row["sigma_ratio_vs_option"] = 1.0
                row["delta_ratio_vs_option"] = 1.0
                continue
            if row.get("sigma_pi_median") is not None:
                row["sigma_ratio_vs_option"] = row["sigma_pi_median"] / option["sigma_pi_median"]
            if row.get("delta_abs_mean") is not None:
                row["delta_ratio_vs_option"] = row["delta_abs_mean"] / option["delta_abs_mean"]
    return out


def _facet_summary_rows(per_facet: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"facet": facet, **dict(row)}
        for facet, row in sorted(dict(per_facet.get("facets", {})).items())
    ]


def _diagnostics(idata: Any, stack: Mapping[str, Any]) -> dict[str, Any]:
    az = stack["az"]
    var_names = ["mu_beta", "sigma_beta", "mu_log_sigma", "tau_log_sigma", "sigma_delta", "theta"]
    summary = az.summary(idata, var_names=var_names)
    rhat = [_float(value) for value in summary.get("r_hat", [])]
    ess_bulk = [_float(value) for value in summary.get("ess_bulk", [])]
    ess_tail = [_float(value) for value in summary.get("ess_tail", [])]
    try:
        diverging = int(idata.sample_stats["diverging"].sum().values)
    except Exception:
        diverging = 0
    return {
        "max_rhat": _max_finite(rhat),
        "min_ess_bulk": _min_finite(ess_bulk),
        "min_ess_tail": _min_finite(ess_tail),
        "diverging": diverging,
        "n_chains": int(idata.posterior.sizes.get("chain", 0)),
        "n_draws": int(idata.posterior.sizes.get("draw", 0)),
    }


def _fit_input_summary(rows: Sequence[Mapping[str, Any]], *, label: str) -> dict[str, Any]:
    by_facet_dataset = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["facet"]), str(row["dataset"]))].append(row)
    for (facet, dataset), group in sorted(grouped.items()):
        by_facet_dataset.append(
            {
                "facet": facet,
                "dataset": dataset,
                "n_rows": len(group),
                "n_items": len({str(row["item_key"]) for row in group}),
                "n_models": len({str(row["model"]) for row in group}),
            }
        )
    return {
        "schema_version": 1,
        "outcome": label,
        "n_rows": len(rows),
        "n_items": len({str(row["item_key"]) for row in rows}),
        "n_models": len({str(row["model"]) for row in rows}),
        "facets": sorted({str(row["facet"]) for row in rows}),
        "datasets": sorted({str(row["dataset"]) for row in rows}),
        "models": sorted({str(row["model"]) for row in rows}),
        "by_facet_dataset": by_facet_dataset,
    }


def _load_pymc_stack() -> dict[str, Any]:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    try:
        import arviz as az
        import pymc as pm
        import pytensor.tensor as pt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "IRT fitting requires optional dependencies. Install with "
            '`python -m pip install -e ".[irt]"` or `uv pip install -e ".[irt]"`.'
        ) from exc
    return {"az": az, "pm": pm, "pt": pt}


def _load_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "IRT fitting requires numpy from the optional `irt` extra. Install with "
            '`python -m pip install -e ".[irt]"`.'
        ) from exc
    return np


def _posterior_mean(idata: Any, name: str) -> Any:
    return idata.posterior[name].mean(dim=("chain", "draw")).values


def _posterior_quantile(idata: Any, name: str, q: float) -> Any:
    return idata.posterior[name].quantile(q, dim=("chain", "draw")).values


def _first_row_by_item(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out = {}
    for row in rows:
        out.setdefault(str(row["item_key"]), row)
    return out


def _coerce_binary(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if int(value) in {0, 1}:
            return int(value)
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "correct"}:
        return 1
    if text in {"0", "false", "no", "incorrect"}:
        return 0
    return None


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _max_finite(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return max(finite) if finite else None


def _min_finite(values: Sequence[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return min(finite) if finite else None


class _SourceTrial:
    def __init__(
        self,
        *,
        facet: str,
        dataset: str,
        model: str,
        item_id: str,
        ordering_idx: int,
        permutation: tuple[int, ...],
        answer_normalized: str | None,
        gold_normalized: str | None,
        correct: bool | None,
        score_kind: str | None,
    ) -> None:
        self.facet = facet
        self.dataset = dataset
        self.model = model
        self.item_id = item_id
        self.ordering_idx = ordering_idx
        self.permutation = permutation
        self.answer_normalized = answer_normalized
        self.gold_normalized = gold_normalized
        self.correct = correct
        self.score_kind = score_kind

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.facet, self.dataset, self.model, self.item_id)


def _source_trial(record: Mapping[str, Any] | TrialRecord) -> _SourceTrial:
    trial = record if isinstance(record, TrialRecord) else TrialRecord.from_mapping(dict(record))
    answer = _clean_text(trial.answer_normalized)
    gold = _clean_text(trial.gold_normalized)
    return _SourceTrial(
        facet=trial.facet,
        dataset=trial.dataset,
        model=trial.model or "",
        item_id=trial.item_id,
        ordering_idx=trial.ordering_idx,
        permutation=trial.permutation,
        answer_normalized=answer,
        gold_normalized=gold,
        correct=trial.correct,
        score_kind=trial.score_kind,
    )


def _modal_answers(
    grouped: Mapping[tuple[str, str, str, str], Sequence[_SourceTrial]],
) -> tuple[dict[tuple[str, str, str, str], str], set[tuple[str, str, str, str]]]:
    modal_by_key = {}
    tied_keys = set()
    for key, trials in grouped.items():
        counts = Counter(trial.answer_normalized for trial in trials if trial.answer_normalized)
        if not counts:
            continue
        ranked = counts.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            tied_keys.add(key)
            continue
        modal_by_key[key] = str(ranked[0][0])
    return modal_by_key, tied_keys


def _irt_row(
    trial: _SourceTrial,
    *,
    outcome: str,
    value: bool,
    modal_answer: str | None,
    n_orderings: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "outcome": outcome,
        "outcome_value": int(bool(value)),
        "facet": trial.facet,
        "dataset": trial.dataset,
        "model": trial.model,
        "item_id": trial.item_id,
        "item_key": "::".join([trial.facet, trial.dataset, trial.item_id]),
        "ordering_idx": trial.ordering_idx,
        "permutation_json": json.dumps(list(trial.permutation), separators=(",", ":")),
        "answer_normalized": trial.answer_normalized,
        "gold_normalized": trial.gold_normalized,
        "correct": trial.correct,
        "modal_answer": modal_answer,
        "n_orderings_in_item": n_orderings,
        "score_kind": trial.score_kind,
    }


def _source_trial_sort_key(trial: _SourceTrial) -> tuple[str, str, str, str, int]:
    return (trial.facet, trial.dataset, trial.model, trial.item_id, trial.ordering_idx)


def _group_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["outcome"]),
                str(row["facet"]),
                str(row["dataset"]),
                str(row["model"]),
            )
        ].append(row)
    out = []
    for (outcome, facet, dataset, model), group in sorted(grouped.items()):
        values = [int(row["outcome_value"]) for row in group]
        out.append(
            {
                "outcome": outcome,
                "facet": facet,
                "dataset": dataset,
                "model": model,
                "n_rows": len(group),
                "n_item_groups": len({str(row["item_id"]) for row in group}),
                "mean_outcome": sum(values) / len(values) if values else None,
            }
        )
    return out


def _compact_released_status(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "release": summary["release"],
        "model_family": summary["model_family"],
        "n_input_trials": summary["input_summary"]["n_trials"],
        "n_input_items": summary["input_summary"]["n_items"],
        "facets": summary["input_summary"]["facets"],
        "modal_diagnostics": summary["outcomes"]["modal"]["diagnostics"],
        "correct_diagnostics": summary["outcomes"]["correct"]["diagnostics"],
    }


def _clean_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
