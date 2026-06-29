"""Facet-Probe public API."""

from facet_probe.facets import FacetSpec, get_facet, sample_permutations
from facet_probe.hf_inspect import HFInspection, build_hf_inspection
from facet_probe.manifests import audit_item_from_mapping, trial_manifest_rows
from facet_probe.metrics import audit_records, item_metrics, summarize_groups
from facet_probe.providers import ProviderSpec, provider_env_status
from facet_probe.reports import build_evaluation_report, write_evaluation_report
from facet_probe.schema import AuditItem, Component, TrialRecord
from facet_probe.scoring import normalize_answer, parse_answer_letter, score_answer
from facet_probe.validation import ValidationReport, validate_audit_items

__all__ = [
    "AuditItem",
    "Component",
    "FacetSpec",
    "HFInspection",
    "ProviderSpec",
    "TrialRecord",
    "ValidationReport",
    "audit_records",
    "audit_item_from_mapping",
    "build_evaluation_report",
    "build_hf_inspection",
    "get_facet",
    "item_metrics",
    "normalize_answer",
    "parse_answer_letter",
    "sample_permutations",
    "score_answer",
    "summarize_groups",
    "provider_env_status",
    "trial_manifest_rows",
    "validate_audit_items",
    "write_evaluation_report",
]

__version__ = "0.0.1"
