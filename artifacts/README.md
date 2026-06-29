# Release Artifacts

These are sanitized public artifacts for the initial Facet-Probe release.

- `paper/`: main table values and dataset summaries.
- `screens/`: public image-set position-reference screen with upstream content removed.
- `odi/`: compact screened ODI summaries and per-item parameter summaries.
- `robustness/`: screened decoder-noise vs ordering decomposition.
- `mitigation/`: screened cost-Pareto, CTA, and think-budget mitigation summaries.
- `provenance/`: paper-finalization audit notes linking artifacts to source runs.

`../configs/release_artifacts.yaml` is the authoritative manifest for this
folder. It maps each shipped file to the paper table, figure, appendix section,
or provenance note it supports.

`reproducibility_matrix.md` is the human-readable version of that mapping, with
the verification command for each result family.

The raw internal experiment tree, upstream dataset content, provider caches, and
API credentials are not included.

To audit this folder in a public clone:

```bash
python scripts/audit_release.py --offline-only
```

To audit against the manifest and local paper workspace:

```bash
python scripts/audit_release.py --source-root /path/to/EMNLP_2026 --arxiv-zip /path/to/paper.zip
```
