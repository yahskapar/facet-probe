# Release Notes

## 0.0.1 Initial Public Code and Evaluation Artifacts Release

- Adds the `facet_probe` Python package.
- Adds paper-facing configs for datasets, models, facets, ordering grammar, and artifacts.
- Ships sanitized aggregate artifacts for the arXiv release.
- Includes screened image-set classification IDs without upstream dataset content.
- Includes compact ODI summaries and excludes multi-gigabyte posterior traces.
- Includes compact Figure 4 CTA and think-budget mitigation summaries.
- Adds CLI commands for listing specs, generating deterministic permutations, building bulk trial manifests, validating normalized items, inspecting HuggingFace metadata, checking provider environment variables, auditing JSONL records, writing report artifacts, and verifying artifacts.
- Adds `scripts/audit_release.py` for manifest coverage, byte-for-byte artifact regeneration, arXiv-source grounding, sanitization, and credential scans.
- Adds `setup.sh` and `environment.yml` so users can install with either `conda` or `uv`.
