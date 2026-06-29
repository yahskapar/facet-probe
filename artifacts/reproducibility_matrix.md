# Reproducibility Matrix

This matrix maps the public Facet-Probe release artifacts to the paper results
they support. It is meant to be read alongside `configs/release_artifacts.yaml`,
which is the machine-readable manifest.

| Paper result | Public artifacts | Source derivation | Verification |
|---|---|---|---|
| Main Figure 2 and Table 1 per-facet flip rates | `paper/per_facet_per_model.csv`, `paper/panel_means.csv`, `paper/screened_panel_values.json`, `provenance/screened_panel_values.md` | `EMNLP_2026/arxiv_finalization/derived/screened_panel_values.json` | `facet-probe verify-artifacts`; `python scripts/audit_release.py --source-root ... --arxiv-zip ...` |
| Main Table 2 ODI decomposition | `odi/facet_decomposition.csv`, `odi/irt_v4_modal_per_facet_summary.json`, `odi/irt_v4_modal_diagnostics.json`, `provenance/screened_odi_fit_audit.md` | screened modal ODI fit under `MMIOS/results/experiments/irt_v6_screened5_modal_*` | `facet-probe verify-artifacts`; release audit byte-for-byte materialization check |
| Appendix ODI posterior intervals | `odi/posterior_intervals.csv`, `provenance/odi_posterior_interval_audit.md` | screened modal ODI posterior summaries from the finalization audit | release audit arXiv-source grounding and artifact manifest checks |
| Image-set screen and screened image rates | `screens/imageset_position_reference_screen.csv`, `screens/imageset_position_reference_screen_summary.json`, `paper/image_set_screened_rates.csv`, `odi/screen_filter_metadata.json` | `MMIOS/data/screens/imageset_position_reference_screen.json`, sanitized for public release | release audit screen row-count, column-boundary, and compact-cell checks |
| Mixed-modality sem-flip cells | `paper/mixed_modality_cells.csv`, `paper/screened_panel_values.json` | final screened panel values with LLM-judge sem-flip labels | `facet-probe verify-artifacts`; release audit arXiv-source grounding |
| Appendix robustness and decoder-noise decomposition | `robustness/decoder_decomp_screened_by_facet.csv`, `robustness/decoder_decomp_screened.json`, `provenance/decoder_decomp_screened_audit.md` | `MMIOS/results/decoder_decomp_screened.*` after the D1 image screen | release audit materialization and manifest checks |
| Main Table 4 and appendix mitigation cost-Pareto results | `mitigation/policy_means.csv`, `mitigation/mitigation_screened_values.json`, `mitigation/mitigation_menu_summary.json`, `mitigation/pareto_frontier_points.json`, `provenance/mitigation_screened_values.md` | `EMNLP_2026/arxiv_finalization/derived/mitigation_screened_values.json` and mitigation-menu experiment summaries | release audit materialization and manifest checks |
| Main Figure 4 and appendix CTA/think-budget mitigation results | `mitigation/cta_flip_summary.csv`, `mitigation/think_budget_sweep.csv`, `provenance/cta_think_budget_summary.md` | compact summaries recomputed from training-free Gemini JSONL outputs under `MMIOS/results/experiments/` | release audit materialization and arXiv-value checks |

## Reproducibility Commands

For a public clone:

```bash
facet-probe verify-artifacts
python scripts/audit_release.py --offline-only
python -m pytest
```

For a clone next to the full paper workspace:

```bash
python scripts/materialize_release_artifacts.py --source-root ../
python scripts/audit_release.py --source-root ../ --arxiv-zip ../arxiv_latest_post_finalization/paper.zip
```

The full release audit checks manifest coverage, exact regeneration of
materialized artifacts, arXiv-source grounding, image-screen sanitization,
credential scans, and conda/uv setup dry-runs.
