# ArXiv Source Audit

This note records the `v0.0.1` audit of Facet-Probe against the latest arXiv
source bundle used for the initial public release.

## Source Bundle

- Archive: latest arXiv source ZIP used for the `v0.0.1` release audit.
- SHA-256: `2a432c522089dc7d899555b001461479f050c96600779a608236bd6b4061b357`
- Source contents inspected: 24 files, including `main.tex`, 6 main-section
  TeX files, 10 supplementary TeX files, 4 PDFs, ACL style files, and
  `references.bib`.
- Latest file timestamp in bundle: `2026-06-27 17:32:58`.

The bundle was extracted and read before this audit. The public repo does not
store the arXiv source itself; it stores this audit note, artifact manifests,
and checks that can be run against a local copy of the source archive.

## Verification Command

Run one of:

```bash
python scripts/audit_release.py --arxiv-zip path/to/arxiv_source.zip
python scripts/audit_release.py --arxiv-source path/to/extracted/arxiv_source
```

The source-grounded audit verifies the public compact artifacts against the
paper source for the rows below.

## Passed Source-To-Artifact Checks

| ArXiv result or promise | Public evidence | Check status |
|---|---|---|
| Code and audit artifacts are released for the paper | `README.md`, `pyproject.toml`, `CITATION.cff`, this artifact bundle | Pass |
| 18-model panel, access window, model IDs, and generation settings | `configs/models.yaml`, `artifacts/paper/per_facet_per_model.csv` | Pass |
| 12 main datasets, audited N, licenses, splits, and primary facets | `configs/datasets.yaml`, `artifacts/paper/dataset_summary.csv` | Pass |
| Five main facets and `K=6`, seed-42 ordering grammar | `configs/facets.yaml`, `configs/orderings.yaml`, `src/facet_probe/facets.py`, `tests/test_facets.py` | Pass |
| Main Figure 1 capability correlations and within-family scaling summaries | `artifacts/paper/capability_summary.csv`, `artifacts/odi/irt_v4_correct_theta.json`, `artifacts/provenance/screened_odi_fit_audit.md` | Pass |
| Main Figure 2 / Table 1 per-facet and per-model flip rates | `artifacts/paper/per_facet_per_model.csv`, `artifacts/paper/panel_means.csv`, `artifacts/paper/screened_panel_values.json` | Pass |
| Image-set position-reference screen and clean Mantis/MedFrameQA counts | `artifacts/screens/`, `artifacts/paper/image_set_screened_rates.csv`, `artifacts/odi/screen_filter_metadata.json` | Pass |
| Main Table 2 ODI facet decomposition | `artifacts/odi/facet_decomposition.csv`, `artifacts/odi/irt_v4_modal_per_facet_summary.json`, `artifacts/provenance/screened_odi_fit_audit.md` | Pass |
| Appendix ODI posterior intervals and diagnostics | `artifacts/odi/posterior_intervals.csv`, `artifacts/odi/*diagnostics.json`, `artifacts/provenance/odi_posterior_interval_audit.md` | Pass |
| Robustness appendix decoder-noise and deployment-temperature summaries | `artifacts/robustness/decoder_decomp_screened_by_facet.csv`, `artifacts/robustness/decoder_decomp_screened.json` | Pass |
| Main Q6 calibration and mechanism-classification diagnostics | `artifacts/diagnostics/calibration_mechanism_summary.csv`, `artifacts/diagnostics/llm_judge_validation.csv` | Pass |
| Appendix demoted facets and tool-description null/stress cells | `artifacts/paper/additional_facet_results.csv`, `configs/facets.yaml`, `configs/datasets.yaml` | Pass |
| Main Table 4 cost-Pareto mitigation means | `artifacts/mitigation/policy_means.csv`, `artifacts/mitigation/mitigation_screened_values.json` | Pass |
| Main Figure 4 CTA and think-budget cells | `artifacts/mitigation/cta_flip_summary.csv`, `artifacts/mitigation/think_budget_sweep.csv` | Pass |
| Mixed-modality sem-flip cells and judge caveat | `artifacts/paper/mixed_modality_cells.csv`, `artifacts/paper/screened_panel_values.json`, `docs/artifacts.md` | Pass |
| Public release boundary for larger planned artifacts | `docs/artifacts.md`, `configs/release_artifacts.yaml`, `README.md` | Pass |

## Deferred Full-Release Items

The arXiv source uses planned-full-release language for several larger artifact
families. They are not required for the `v0.0.1` compact public artifact audit,
but they remain explicit follow-on work for `v1.0.0`:

- full permutation manifests / permutation indices,
- normalized trial outputs and raw model outputs,
- aggregation scripts for every paper table and figure,
- full prompt templates and judge prompt templates,
- production dataset loaders for the 12 paper datasets and more model-adapter examples,
- per-cell flip/OSI tables, bootstrap intervals, and other diagnostic CSVs,
- full posterior traces and per-parameter posterior CSVs where storage permits,
- per-item judge labels, full judge prompts, and calibration-dialog transcripts.

These are deferred because they require additional provider-output review,
dataset-license review, storage planning, or content-redaction work. The
`v0.0.1` release intentionally ships compact, sanitized artifacts that reproduce
the reported aggregate results without redistributing upstream dataset content
or raw provider responses.
