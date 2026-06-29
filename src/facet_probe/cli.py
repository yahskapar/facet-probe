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
from facet_probe.manifests import trial_manifest_rows
from facet_probe.metrics import audit_records, read_jsonl, summarize_groups, write_csv, write_json
from facet_probe.providers import PROVIDERS, provider_env_status
from facet_probe.reports import write_evaluation_report
from facet_probe.validation import validate_audit_items


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
    names = sorted(PROVIDERS) if "all" in args.providers else args.providers
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
    p.add_argument("--providers", nargs="+", default=["all"], choices=["all", *sorted(PROVIDERS)])
    p.set_defaults(func=_cmd_check_env)

    p = sub.add_parser("verify-artifacts", help="Check included release artifact consistency.")
    p.set_defaults(func=_cmd_verify_artifacts)

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
