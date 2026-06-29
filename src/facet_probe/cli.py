"""Command line interface for Facet-Probe."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from facet_probe.artifacts import verify_release_artifacts
from facet_probe.datasets import list_paper_datasets
from facet_probe.facets import FACETS, get_facet, sample_permutations
from facet_probe.hf_inspect import inspect_hf_dataset
from facet_probe.irt import (
    released_irt_summary,
    write_irt_fit,
    write_irt_input,
    write_released_irt_summary,
)
from facet_probe.judging import judge_mixed_trials
from facet_probe.manifests import trial_manifest_rows
from facet_probe.metrics import audit_records, read_jsonl, summarize_groups, write_csv, write_json
from facet_probe.profiles import (
    EvaluationProfile,
    ModelProfile,
    judge_profile,
    model_profile,
    paper_profile,
)
from facet_probe.providers import PROVIDERS, provider_env_status
from facet_probe.reports import write_evaluation_report
from facet_probe.runner import execute_profile
from facet_probe.validation import validate_audit_items

_PUBLIC_PROVIDERS = tuple(name for name in sorted(PROVIDERS) if name != "mock")


def _cmd_list_facets(_args: argparse.Namespace) -> int:
    for spec in FACETS.values():
        marker = "main" if spec.main_paper else "demoted"
        print(f"{spec.name}\t{marker}\t{spec.unit}\t{spec.score_kind}")
    return 0


def _cmd_list_datasets(_args: argparse.Namespace) -> int:
    for spec in list_paper_datasets():
        facets = ",".join(spec.primary_facets)
        print(f"{spec.name}\t{spec.hf_repo}\t{spec.split}\tN={spec.audited_n}\t{facets}")
    return 0


def _cmd_make_permutations(args: argparse.Namespace) -> int:
    perms = sample_permutations(
        args.n_components,
        k=args.k,
        seed=args.seed,
        item_id=args.item_id,
        include_canonical=not args.no_canonical,
    )
    for idx, perm in enumerate(perms):
        print(json.dumps({"ordering_idx": idx, "permutation": list(perm)}))
    return 0


def _cmd_make_manifest(args: argparse.Namespace) -> int:
    get_facet(args.facet)
    items = read_jsonl(args.items_jsonl)
    rows = trial_manifest_rows(
        items,
        facet=args.facet,
        k=args.k,
        seed=args.seed,
        include_ordered_components=args.include_ordered_components,
    )
    _write_jsonl(args.output, rows)
    return 0


def _cmd_validate_items(args: argparse.Namespace) -> int:
    items = read_jsonl(args.items_jsonl)
    report = validate_audit_items(items, facet=args.facet, k=args.k)
    obj = report.to_dict()
    print(json.dumps(obj, indent=2, sort_keys=True))
    if args.report_json:
        write_json(args.report_json, obj)
    return 0 if report.ok else 1


def _cmd_audit_jsonl(args: argparse.Namespace) -> int:
    records = read_jsonl(args.path)
    summary = audit_records(records, label=args.label or Path(args.path).stem)
    group_rows = summarize_groups(records)
    print(json.dumps(asdict(summary), indent=2))
    if args.summary_json:
        write_json(args.summary_json, asdict(summary))
    if args.group_csv:
        write_csv(args.group_csv, group_rows)
    return 0


def _cmd_make_report(args: argparse.Namespace) -> int:
    records = read_jsonl(args.path)
    label = args.label or Path(args.path).stem
    paths = write_evaluation_report(
        args.output_dir,
        records,
        label=label,
        group_by=tuple(args.group_by),
        include_items=not args.no_item_csv,
    )
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
    return 0


def _cmd_check_env(args: argparse.Namespace) -> int:
    names = list(_PUBLIC_PROVIDERS) if "all" in args.providers else args.providers
    statuses = [provider_env_status(name) for name in names]
    for status in statuses:
        print(json.dumps(status, sort_keys=True))
    return 0 if all(bool(status["ok"]) for status in statuses) else 1


def _cmd_verify_artifacts(_args: argparse.Namespace) -> int:
    checks = verify_release_artifacts()
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        print(f"{status}\t{check.name}\t{check.detail}")
    return 0 if all(check.ok for check in checks) else 1


def _cmd_irt_summary(args: argparse.Namespace) -> int:
    if args.output_dir:
        status = write_released_irt_summary(args.output_dir)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    print(json.dumps(released_irt_summary(), indent=2, sort_keys=True))
    return 0


def _cmd_irt_export(args: argparse.Namespace) -> int:
    records = read_jsonl(args.trials_jsonl)
    status = write_irt_input(records, args.output_dir, outcomes=tuple(args.outcomes))
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _cmd_irt_fit(args: argparse.Namespace) -> int:
    status = write_irt_fit(
        args.irt_input,
        args.output_dir,
        outcome=args.outcome,
        n_chains=args.n_chains,
        n_draws=args.n_draws,
        n_tune=args.n_tune,
        target_accept=args.target_accept,
        nuts_sampler=args.nuts_sampler,
        chain_method=args.chain_method,
        seed=args.seed,
        limit_items_per_facet=args.limit_items_per_facet,
        dry_run=args.dry_run,
        save_idata=args.save_idata,
        progressbar=args.progressbar,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _cmd_inspect_hf(args: argparse.Namespace) -> int:
    inspection = inspect_hf_dataset(
        args.dataset,
        config=args.config,
        split=args.split,
        revision=args.revision,
        sample=args.sample,
        streaming=not args.no_streaming,
    )
    obj = inspection.to_dict()
    if args.output:
        write_json(args.output, obj)

    spec_obj = {"datasets": inspection.starter_dataset_spec}
    if args.spec_output:
        Path(args.spec_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.spec_output).write_text(
            yaml.safe_dump(spec_obj, sort_keys=False),
            encoding="utf-8",
        )

    if args.emit_spec:
        print(yaml.safe_dump(spec_obj, sort_keys=False).rstrip())
    elif not args.output:
        print(json.dumps(obj, indent=2, sort_keys=True))
    return 0


def _cmd_paper_run(args: argparse.Namespace) -> int:
    profile = paper_profile(
        config_dir=args.config_dir,
        model_config=args.model_config,
        models=args.models,
        datasets=args.datasets,
        k=args.k,
        seed=args.seed,
        name=args.name,
    )

    extra_models = []
    for hf_model in args.hf_model:
        extra_models.append(model_profile("huggingface", hf_model))
    if args.mock_model:
        extra_models.append(model_profile("mock", args.mock_model))
    if args.provider or args.api_model:
        if not args.provider or not args.api_model:
            raise ValueError("--provider and --api-model must be supplied together")
        generation = {"endpoint_url": args.endpoint_url} if args.endpoint_url else None
        extra_models.append(
            model_profile(
                args.provider,
                args.api_model,
                api_key_env=args.api_key_env,
                generation=generation,
            )
        )
    if extra_models:
        if args.models:
            profile = profile.add_models(*extra_models)
        else:
            profile = profile.only_models(*extra_models)

    plan = _paper_run_plan(profile, prepare_only=args.prepare_only)
    if args.output_dir:
        files = _write_paper_run_dir(Path(args.output_dir), profile, plan)
        plan["files"] = files
    elif not args.prepare_only:
        raise ValueError("--output-dir is required unless --prepare-only is set")

    if not args.prepare_only:
        run_status = execute_profile(
            profile,
            args.output_dir,
            items_jsonl=args.items_jsonl,
            item_facet=args.item_facet,
            limit_items=args.limit_items,
            limit_trials=args.limit_trials,
            streaming=not args.no_streaming,
            include_raw_outputs=not args.no_raw_outputs,
            max_new_tokens=args.max_new_tokens,
            allow_partial=args.allow_partial,
        )
        plan = {
            "status": run_status["status"],
            "profile": profile.to_dict(),
            "provider_status": profile.provider_status(),
            "summary": _load_json(run_status["files"]["summary"]),
            "files": {**plan.get("files", {}), **run_status["files"]},
            "skipped_row_counts": run_status["skipped_row_counts"],
            "shortfalls": run_status["shortfalls"],
        }
        if args.judge_mixed:
            records = read_jsonl(run_status["files"]["trials"])
            judge = _judge_model_from_args(
                args,
                provider_attr="judge_provider",
                api_model_attr="judge_api_model",
                api_key_env_attr="judge_api_key_env",
                endpoint_url_attr="judge_endpoint_url",
            )
            _raise_if_missing_env(judge, role="judge")
            judge_output_dir = args.judge_output_dir or str(
                Path(args.output_dir) / "mixed_semantic_judge"
            )
            judge_status = judge_mixed_trials(
                records,
                output_dir=judge_output_dir,
                judge_model=judge,
                max_new_tokens=args.judge_max_new_tokens,
                limit_items=args.judge_limit_items,
            )
            plan["mixed_semantic_judge"] = judge_status
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


def _cmd_judge_mixed(args: argparse.Namespace) -> int:
    records = read_jsonl(args.trials_jsonl)
    judge = None
    if not args.mock_judge:
        judge = _judge_model_from_args(args)
        _raise_if_missing_env(judge, role="judge")
    output_dir = args.output_dir or str(Path(args.trials_jsonl).parent / "mixed_semantic_judge")
    status = judge_mixed_trials(
        records,
        output_dir=output_dir,
        judge_model=judge,
        mock_judge=args.mock_judge,
        max_new_tokens=args.max_new_tokens,
        limit_items=args.limit_items,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def _judge_model_from_args(
    args: argparse.Namespace,
    *,
    provider_attr: str = "provider",
    api_model_attr: str = "api_model",
    api_key_env_attr: str = "api_key_env",
    endpoint_url_attr: str = "endpoint_url",
) -> ModelProfile:
    provider = getattr(args, provider_attr)
    api_model = getattr(args, api_model_attr)
    api_key_env = getattr(args, api_key_env_attr)
    endpoint_url = getattr(args, endpoint_url_attr)
    if provider or api_model:
        if not provider or not api_model:
            raise ValueError("--provider and --api-model must be supplied together")
        generation = {"endpoint_url": endpoint_url} if endpoint_url else None
        return model_profile(
            provider,
            api_model,
            api_key_env=api_key_env,
            generation=generation,
        )
    return judge_profile(
        config_dir=args.config_dir,
        judge_config=args.judge_config,
        name=args.judge,
    )


def _raise_if_missing_env(model: ModelProfile, *, role: str) -> None:
    status = model.env_status()
    missing = [
        key
        for key, ok in dict(status.get("required_env", {})).items()
        if not ok
    ]
    if missing:
        raise RuntimeError(
            f"{role} profile {model.name!r} requires environment variable(s): "
            f"{', '.join(missing)}"
        )


def _paper_run_plan(profile: EvaluationProfile, *, prepare_only: bool) -> dict[str, object]:
    status = "prepared" if prepare_only else "ready_to_execute"
    return {
        "status": status,
        "profile": profile.to_dict(),
        "provider_status": profile.provider_status(),
        "next_steps": [
            "inspect run_profile.json, models.jsonl, and datasets.jsonl",
            "rerun without --prepare-only to execute this profile",
            "review manifest.jsonl, trials.jsonl, summary.json, and report/",
        ],
    }


def _write_paper_run_dir(
    output_dir: Path,
    profile: EvaluationProfile,
    plan: dict[str, object],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "run_profile": output_dir / "run_profile.json",
        "provider_status": output_dir / "provider_status.json",
        "models": output_dir / "models.jsonl",
        "datasets": output_dir / "datasets.jsonl",
        "readme": output_dir / "README.md",
    }
    write_json(files["run_profile"], profile.to_dict())
    write_json(files["provider_status"], profile.provider_status())
    _write_jsonl(str(files["models"]), [model.to_dict() for model in profile.models])
    _write_jsonl(str(files["datasets"]), [dataset.to_dict() for dataset in profile.datasets])
    files["readme"].write_text(_paper_run_readme(profile), encoding="utf-8")
    return {name: str(path) for name, path in files.items()}


def _paper_run_readme(profile: EvaluationProfile) -> str:
    return "\n".join(
        [
            "# Facet-Probe Paper Run",
            "",
            "This directory contains a Facet-Probe paper-benchmark run.",
            "",
            f"- Profile: `{profile.name}`",
            f"- Models: {len(profile.models)}",
            f"- Datasets: {len(profile.datasets)}",
            f"- K orderings: {profile.k_orderings}",
            f"- Seed: {profile.seed}",
            "",
            "Profile files are written before inference. Completed runs also include:",
            "",
            "- `manifest.jsonl`: one row per item/order facet trial.",
            "- `trials.jsonl`: normalized model outputs and scores.",
            "- `summary.json` and `group_summary.csv`: aggregate metrics.",
            "- `report/`: summary, group, item, and manifest report files.",
            "- `irt_input/`: optional modal/correct outcome export when created",
            "  with `facet-probe irt-export trials.jsonl --output-dir irt_input`.",
            "- `irt_fit/`: optional Bayesian ODI/IRT fit outputs when created",
            "  with `facet-probe irt-fit irt_input/irt_input_trials.csv --output-dir irt_fit`.",
            "- `run_status.json`: counts, output paths, and skipped-row diagnostics.",
            "- `mixed_semantic_judge/`: semantic-equivalence judge outputs when",
            "  `paper-run --judge-mixed` was used.",
            "",
            "You can recompute summaries from `trials.jsonl` with:",
            "",
            "```bash",
            "facet-probe audit-jsonl trials.jsonl \\",
            "  --summary-json summary.json \\",
            "  --group-csv group_summary.csv",
            "facet-probe make-report trials.jsonl --output-dir report",
            "```",
            "",
        ]
    )


def _load_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path: str | None, rows: list[dict[str, object]]) -> None:
    if path is None:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="facet-probe")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-facets", help="List registered ordering facets.")
    p.set_defaults(func=_cmd_list_facets)

    p = sub.add_parser("list-datasets", help="List datasets used in the paper audit.")
    p.set_defaults(func=_cmd_list_datasets)

    p = sub.add_parser("make-permutations", help="Emit deterministic K-ordering manifest rows.")
    p.add_argument("--n-components", type=int, required=True)
    p.add_argument("--item-id", default="")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-canonical", action="store_true")
    p.set_defaults(func=_cmd_make_permutations)

    p = sub.add_parser(
        "make-manifest",
        help="Build K-ordering trial manifest from AuditItem JSONL.",
    )
    p.add_argument("items_jsonl")
    p.add_argument("--facet", required=True)
    p.add_argument("--output")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-ordered-components", action="store_true")
    p.set_defaults(func=_cmd_make_manifest)

    p = sub.add_parser("validate-items", help="Validate normalized AuditItem JSONL.")
    p.add_argument("items_jsonl")
    p.add_argument("--facet")
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--report-json")
    p.set_defaults(func=_cmd_validate_items)

    p = sub.add_parser("audit-jsonl", help="Compute flip rate and OSI from trial JSONL.")
    p.add_argument("path")
    p.add_argument("--label")
    p.add_argument("--summary-json")
    p.add_argument("--group-csv")
    p.set_defaults(func=_cmd_audit_jsonl)

    p = sub.add_parser("make-report", help="Write summary/group/item evaluation artifacts.")
    p.add_argument("path")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label")
    p.add_argument("--group-by", nargs="+", default=["facet", "dataset", "model"])
    p.add_argument("--no-item-csv", action="store_true")
    p.set_defaults(func=_cmd_make_report)

    p = sub.add_parser("check-env", help="Check provider env vars without printing secret values.")
    p.add_argument("--providers", nargs="+", default=["all"], choices=["all", *_PUBLIC_PROVIDERS])
    p.set_defaults(func=_cmd_check_env)

    p = sub.add_parser("verify-artifacts", help="Check included release artifact consistency.")
    p.set_defaults(func=_cmd_verify_artifacts)

    p = sub.add_parser(
        "irt-summary",
        help="Inspect or write the released ODI/IRT artifact summary bundle.",
    )
    p.add_argument("--output-dir", help="Write summary files and copy compact ODI artifacts.")
    p.set_defaults(func=_cmd_irt_summary)

    p = sub.add_parser(
        "irt-export",
        help="Export trial JSONL to modal/correct IRT-compatible outcome rows.",
    )
    p.add_argument("trials_jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--outcomes",
        nargs="+",
        default=["modal", "correct"],
        choices=["modal", "correct"],
    )
    p.set_defaults(func=_cmd_irt_export)

    p = sub.add_parser(
        "irt-fit",
        help="Fit the public Bayesian ODI/IRT model from exported outcome rows.",
    )
    p.add_argument("irt_input", help="CSV or JSONL produced by `facet-probe irt-export`.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--outcome", default="modal", choices=["modal", "correct", "both"])
    p.add_argument("--n-chains", type=int, default=4)
    p.add_argument("--n-draws", type=int, default=1500)
    p.add_argument("--n-tune", type=int, default=1500)
    p.add_argument("--target-accept", type=float, default=0.95)
    p.add_argument("--nuts-sampler", default="numpyro", choices=["numpyro", "pymc", "blackjax"])
    p.add_argument(
        "--chain-method",
        default="parallel",
        choices=["parallel", "sequential", "vectorized"],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit-items-per-facet",
        type=int,
        help="Fit only the first N item groups per facet for quick validation runs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize fit inputs without importing PyMC or sampling.",
    )
    p.add_argument(
        "--save-idata",
        action="store_true",
        help="Also save the full ArviZ InferenceData NetCDF trace.",
    )
    p.add_argument("--progressbar", action="store_true", help="Show PyMC sampling progress.")
    p.set_defaults(func=_cmd_irt_fit)

    p = sub.add_parser("inspect-hf", help="Inspect a HuggingFace dataset and suggest facets.")
    p.add_argument("dataset")
    p.add_argument("--config")
    p.add_argument("--split")
    p.add_argument("--revision")
    p.add_argument("--sample", type=int, default=20)
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--emit-spec", action="store_true")
    p.add_argument("--output")
    p.add_argument("--spec-output")
    p.set_defaults(func=_cmd_inspect_hf)

    p = sub.add_parser("paper-run", help="Run or prepare a paper benchmark profile.")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--model-config")
    p.add_argument("--models", nargs="+")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--hf-model", action="append", default=[])
    p.add_argument(
        "--mock-model",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--provider", choices=_PUBLIC_PROVIDERS)
    p.add_argument("--api-model")
    p.add_argument("--api-key-env")
    p.add_argument("--endpoint-url")
    p.add_argument(
        "--judge-mixed",
        action="store_true",
        help="Run the configured mixed-modality semantic judge after inference.",
    )
    p.add_argument("--judge-config", help="YAML file containing named judge profiles.")
    p.add_argument(
        "--judge",
        default="mixed-semantic-primary",
        help="Named judge profile to use with --judge-mixed.",
    )
    p.add_argument("--judge-output-dir")
    p.add_argument("--judge-provider", choices=_PUBLIC_PROVIDERS)
    p.add_argument("--judge-api-model")
    p.add_argument("--judge-api-key-env")
    p.add_argument("--judge-endpoint-url")
    p.add_argument("--judge-max-new-tokens", type=int)
    p.add_argument("--judge-limit-items", type=int)
    p.add_argument("--k", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--name", default="facet-probe-paper-v0.0.1")
    p.add_argument("--output-dir")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--items-jsonl", help="Run normalized AuditItem JSONL instead of HF datasets.")
    p.add_argument("--item-facet", default="option_order")
    p.add_argument("--limit-items", type=int)
    p.add_argument("--limit-trials", type=int)
    p.add_argument("--no-streaming", action="store_true")
    p.add_argument("--no-raw-outputs", action="store_true")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--allow-partial", action="store_true")
    p.set_defaults(func=_cmd_paper_run)

    p = sub.add_parser(
        "judge-mixed",
        help="Judge mixed-modality free-form outputs for semantic flip.",
    )
    p.add_argument("trials_jsonl")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--judge-config", help="YAML file containing named judge profiles.")
    p.add_argument(
        "--judge",
        default="mixed-semantic-primary",
        help="Named judge profile from configs/models.yaml.",
    )
    p.add_argument("--output-dir")
    p.add_argument(
        "--mock-judge",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--provider", choices=_PUBLIC_PROVIDERS)
    p.add_argument("--api-model")
    p.add_argument("--api-key-env")
    p.add_argument("--endpoint-url")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--limit-items", type=int)
    p.set_defaults(func=_cmd_judge_mixed)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as exc:  # pragma: no cover - human CLI path
        print(f"facet-probe: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
