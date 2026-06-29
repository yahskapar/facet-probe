# Artifact Licenses and Release Boundaries

The repository license is Apache-2.0 for code, configs, docs, and repo-native derived artifacts authored for Facet-Probe.

## Included Public Artifacts

The shipped artifacts are aggregate or derived release files:

- per-facet and per-model flip-rate tables,
- screened image-set item IDs and classifications,
- decoder-noise decomposition summaries,
- mitigation policy summaries,
- ODI per-facet summaries, posterior interval summaries, theta summaries, and compact per-item parameter parquet files.

These artifacts are intended for research reproduction of the paper's reported numbers.

## Not Included

This repo intentionally does not redistribute upstream dataset text, images, tables, or prompts. It also excludes credentials, provider caches, raw 78GB internal experiment trees, and multi-gigabyte posterior traces.

The sanitized image-set screen includes item IDs, classifications, regex flags, and aggregate behavioral diagnostics, but omits question text, choices, and free-text rationales from upstream datasets.

## Third-Party Terms

Using the reproduction configs may require downloading upstream datasets or calling model providers. Those resources are subject to their own licenses and terms, including non-commercial restrictions for MedXpertQA and MedFrameQA.

Closed-source provider outputs and APIs may be subject to provider-specific terms. The arXiv paper pins the closed-source access window to May 4-25, 2026 as the reproducibility anchor.
