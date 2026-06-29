# Public Artifacts

This initial release ships the polished artifacts needed to audit the arXiv numbers without redistributing upstream datasets.

`configs/release_artifacts.yaml` is the authoritative manifest. It lists every
file under `artifacts/` and maps artifacts to paper tables, figures, appendix
claims, or provenance notes.

`artifacts/reproducibility_matrix.md` provides the same correspondence in a
human-readable table with the relevant verification commands.

## Release Boundary

The arXiv source uses "planned full code release" language for several large or
provider-sensitive objects. This initial public artifact release does not ship
raw provider caches, upstream dataset content, full normalized model-output
JSONL, full permutation-index dumps, full prompt text, multi-gigabyte posterior
traces, or full per-cell diagnostic tables unless a file is explicitly listed in
`configs/release_artifacts.yaml`.

Instead, the release ships compact, sanitized artifacts that reproduce the
paper's reported aggregate numbers, plus provenance notes and public checks for
the shipped artifact set. This is OK for `v0.0.1` because the public release is
an audit-and-reproduction bundle for the reported aggregate results, not a
redistribution of upstream datasets or provider responses.

## Planned Full-Release Items

The arXiv source also describes larger follow-on artifacts. For `v0.0.1`, these
are intentionally deferred until provider, dataset-license, storage, and review
constraints are cleared:

- full permutation manifests / permutation indices,
- normalized trial outputs and raw model outputs,
- aggregation scripts for all paper tables and figures,
- prompt templates and judge prompt templates,
- production dataset loaders for the 12 paper datasets and more model-adapter examples,
- per-cell diagnostic tables and bootstrap intervals,
- per-item judge labels and calibration-dialog transcripts.

## Main Tables

- `artifacts/paper/per_facet_per_model.csv`: model by facet flip rates from the final screened paper table.
- `artifacts/paper/panel_means.csv`: panel means for the 5 main facets.
- `artifacts/paper/capability_summary.csv`: Main Figure 1 capability/scaling aggregate checks.
- `artifacts/paper/additional_facet_results.csv`: compact appendix values for demoted facets and the tool-description null/stress cells.
- `artifacts/paper/mixed_modality_cells.csv`: per-model, per-benchmark mixed-modality sem-flip values.
- `artifacts/paper/image_set_screened_rates.csv`: screened Mantis-Eval and MedFrameQA image-set rates by model.
- `artifacts/paper/dataset_summary.csv`: audited datasets, licenses, splits, and primary facets.

## Diagnostics

- `artifacts/diagnostics/calibration_mechanism_summary.csv`: compact Q6 calibration and mechanism-classification values.
- `artifacts/diagnostics/llm_judge_validation.csv`: compact LLM-judge agreement and MMQA gold-anchor validation values.

Per-item judge labels, judge prompts, calibration-dialog transcripts, and raw
model outputs are not shipped in `v0.0.1`; they are listed as expanded-release
items because they require provider-output review and prompt/content redaction.

## Screens

- `artifacts/screens/imageset_position_reference_screen.csv`: sanitized item IDs and screen labels.
- `artifacts/screens/imageset_position_reference_screen_summary.json`: methodology, classification rules, and summary counts.

The non-public human-review screen included full question text, answer choices, and rationales. Those fields are omitted here to avoid redistributing upstream dataset content.

## ODI

- `artifacts/odi/facet_decomposition.csv`: screened modal-outcome ODI values used in the paper.
- `artifacts/odi/posterior_intervals.csv`: appendix posterior interval table.
- `artifacts/odi/*theta.json`: model ability summaries.
- `artifacts/odi/*per_item_params.parquet`: compact per-item posterior parameter summaries.

The multi-gigabyte `idata.nc` and `raw_multitrace.pkl` posterior traces are intentionally excluded.

## Robustness and Mitigation

- `artifacts/robustness/decoder_decomp_screened_by_facet.csv`: screened same-ordering vs cross-ordering decomposition.
- `artifacts/mitigation/policy_means.csv`: screened mitigation policy means.
- `artifacts/mitigation/mitigation_screened_values.json`: full screened mitigation summary values.
- `artifacts/mitigation/cta_flip_summary.csv`: compact baseline-vs-CTA cells for Figure 4a and the MedXpertQA CTA+multi-pass anti-synergy claim.
- `artifacts/mitigation/think_budget_sweep.csv`: compact hard/easy Gemini think-budget sweep cells for Figure 4b.

The uncertainty-clean MedXpertQA CTA subset is discussed in the paper text, but
the item-level subset labels and raw model-output JSONL are not included in this
initial public artifact release.

## Provenance

`artifacts/provenance/` contains compact finalization audit notes for the public
artifact values. Raw run paths and non-public source paths are intentionally
omitted from public artifacts.

## Audit Command

Run:

```bash
facet-probe verify-artifacts
python scripts/audit_release.py --offline-only
python scripts/audit_release.py --arxiv-zip path/to/arxiv_source.zip
```

The audit fails if unmanifested artifacts are present, manifest-listed artifacts
are missing, compact numeric artifacts drift from expected values, the screen
appears to include upstream content fields, or obvious credentials are found in
text files.
