# Release Artifacts

These are sanitized public artifacts for the initial Facet-Probe release.

- `paper/`: main table values, dataset summaries, capability summaries, and demoted-facet summary values.
- `diagnostics/`: compact calibration, mechanism-classification, and LLM-judge validation summaries.
- `screens/`: public image-set position-reference screen with upstream content removed.
- `odi/`: compact screened ODI summaries and per-item parameter summaries.
- `robustness/`: screened decoder-noise vs ordering decomposition.
- `mitigation/`: screened cost-Pareto, CTA, and think-budget mitigation summaries.
- `provenance/`: compact paper-finalization audit notes for public artifact values.

`../configs/release_artifacts.yaml` is the authoritative manifest for this
folder. It maps each shipped file to the paper table, figure, appendix section,
or provenance note it supports.

`reproducibility_matrix.md` is the human-readable version of that mapping, with
the verification command for each result family.

`arxiv_audit.md` records the source-bundle hash and the paper-to-artifact checks
used for the initial public release.

Raw provider-output run trees, upstream dataset content, per-item judge labels,
prompt transcripts, provider caches, and API credentials are not included.

To audit this folder in a public clone:

```bash
facet-probe verify-artifacts
python scripts/audit_release.py --offline-only
python scripts/audit_release.py --arxiv-zip path/to/arxiv_source.zip
```
