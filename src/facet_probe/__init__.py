"""Facet-Probe public API."""

from facet_probe.facets import FacetSpec, get_facet, sample_permutations
from facet_probe.hf_inspect import HFInspection, build_hf_inspection
from facet_probe.judging import judge_mixed_trials, parse_judge_response
from facet_probe.manifests import audit_item_from_mapping, trial_manifest_rows
from facet_probe.metrics import audit_records, item_metrics, summarize_groups
from facet_probe.profiles import (
    DatasetProfile,
    EvaluationProfile,
    ModelProfile,
    closed_source_model_profile,
    dataset_profile_from_hf,
    hf_dataset,
    huggingface_model_profile,
    judge_profile,
    model_profile,
    paper_evaluation_profile,
    paper_profile,
)
from facet_probe.providers import ProviderSpec, provider_env_status
from facet_probe.reports import build_evaluation_report, write_evaluation_report
from facet_probe.runner import (
    ModelResponse,
    RuntimeExample,
    create_model_adapter,
    execute_profile,
    load_runtime_examples,
)
from facet_probe.schema import AuditItem, Component, TrialRecord
from facet_probe.scoring import normalize_answer, parse_answer_letter, score_answer
from facet_probe.templates import (
    content_ref,
    evidence_list_audit_item,
    image_list_audit_item,
    mcq_audit_item,
    mixed_modality_audit_item,
    render_ordered_text_prompt,
)
from facet_probe.validation import ValidationReport, validate_audit_items

__all__ = [
    "AuditItem",
    "Component",
    "DatasetProfile",
    "EvaluationProfile",
    "FacetSpec",
    "HFInspection",
    "ModelProfile",
    "ModelResponse",
    "ProviderSpec",
    "RuntimeExample",
    "TrialRecord",
    "ValidationReport",
    "__version__",
    "audit_records",
    "audit_item_from_mapping",
    "build_evaluation_report",
    "build_hf_inspection",
    "closed_source_model_profile",
    "content_ref",
    "create_model_adapter",
    "dataset_profile_from_hf",
    "evidence_list_audit_item",
    "execute_profile",
    "get_facet",
    "hf_dataset",
    "huggingface_model_profile",
    "image_list_audit_item",
    "item_metrics",
    "judge_mixed_trials",
    "judge_profile",
    "load_runtime_examples",
    "mcq_audit_item",
    "mixed_modality_audit_item",
    "model_profile",
    "normalize_answer",
    "paper_evaluation_profile",
    "paper_profile",
    "parse_judge_response",
    "parse_answer_letter",
    "render_ordered_text_prompt",
    "sample_permutations",
    "score_answer",
    "summarize_groups",
    "provider_env_status",
    "trial_manifest_rows",
    "validate_audit_items",
    "write_evaluation_report",
]

__version__ = "0.0.1"
