# Reproducibility Matrix

This matrix maps the public Facet-Probe release artifacts to the paper results
they support. It is meant to be read alongside `configs/release_artifacts.yaml`,
which is the machine-readable manifest.

| Paper result | Public artifacts | Public verification |
|---|---|---|
| ArXiv source-to-artifact audit | `arxiv_audit.md` | `python scripts/audit_release.py --arxiv-zip path/to/arxiv_source.zip` |
| Main Figure 1 capability and appendix within-family scaling | `paper/capability_summary.csv`, `paper/per_facet_per_model.csv`, `paper/screened_panel_values.json`, `odi/irt_v4_correct_theta.json`, `provenance/screened_odi_fit_audit.md` | `python scripts/audit_release.py --arxiv-zip path/to/arxiv_source.zip` |
| Main Figure 2 and Table 1 per-facet flip rates | `paper/per_facet_per_model.csv`, `paper/panel_means.csv`, `paper/screened_panel_values.json`, `provenance/screened_panel_values.md` | `facet-probe verify-artifacts`; `python scripts/audit_release.py --offline-only` |
| Main Table 2 ODI decomposition | `odi/facet_decomposition.csv`, `odi/irt_v4_modal_per_facet_summary.json`, `odi/irt_v4_modal_diagnostics.json`, `provenance/screened_odi_fit_audit.md` | `facet-probe verify-artifacts`; `python scripts/audit_release.py --offline-only` |
| Appendix ODI posterior intervals | `odi/posterior_intervals.csv`, `provenance/odi_posterior_interval_audit.md` | manifest coverage and provenance-note checks |
| Image-set screen and screened image rates | `screens/imageset_position_reference_screen.csv`, `screens/imageset_position_reference_screen_summary.json`, `paper/image_set_screened_rates.csv`, `odi/screen_filter_metadata.json` | screen boundary checks; `facet-probe verify-artifacts` |
| Mixed-modality sem-flip cells and LLM-judge validation | `paper/mixed_modality_cells.csv`, `paper/screened_panel_values.json`, `diagnostics/llm_judge_validation.csv` | `facet-probe verify-artifacts`; source-grounded audit checks |
| Main Q6 confidence calibration and mechanism diagnostics | `diagnostics/calibration_mechanism_summary.csv`, `diagnostics/llm_judge_validation.csv` | source-grounded audit checks; compact aggregate artifact coverage |
| Appendix additional facets and tool-description null/stress cells | `paper/additional_facet_results.csv`, `configs/facets.yaml`, `configs/datasets.yaml` | source-grounded audit checks; compact aggregate artifact coverage |
| Appendix robustness and decoder-noise decomposition | `robustness/decoder_decomp_screened_by_facet.csv`, `robustness/decoder_decomp_screened.json`, `provenance/decoder_decomp_screened_audit.md` | manifest coverage and provenance-note checks |
| Main Table 4 and appendix mitigation cost-Pareto results | `mitigation/policy_means.csv`, `mitigation/mitigation_screened_values.json`, `mitigation/mitigation_menu_summary.json`, `mitigation/pareto_frontier_points.json`, `provenance/mitigation_screened_values.md` | `facet-probe verify-artifacts`; manifest coverage checks |
| Main Figure 4 and appendix CTA/think-budget mitigation results | `mitigation/cta_flip_summary.csv`, `mitigation/think_budget_sweep.csv`, `provenance/cta_think_budget_summary.md` | `facet-probe verify-artifacts`; manifest coverage checks |

## Reproducibility Commands

For a public clone:

```bash
facet-probe verify-artifacts
python scripts/audit_release.py --offline-only
python scripts/audit_release.py --arxiv-zip path/to/arxiv_source.zip
python -m pytest
```

The public release audit checks manifest coverage, compact numeric artifact
consistency, image-screen sanitization, credential scans, and conda/uv setup
documentation. Raw upstream datasets, provider outputs, local API caches, and
large posterior traces are intentionally omitted from `v0.0.1`. Per-item judge
labels, judge prompts, and calibration-dialog transcripts are likewise deferred
to the expanded artifact release after review.
