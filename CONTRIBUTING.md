# Contributing

Thanks for helping improve Facet-Probe.

## Development Setup

```bash
bash setup.sh uv
source .venv/bin/activate
python -m ruff check .
python -m pytest
facet-probe verify-artifacts
```

Conda users can instead run:

```bash
bash setup.sh conda
conda activate facet-probe
```

When checking paper correspondence, run the source-grounded audit against an
arXiv source ZIP or extracted source directory:

```bash
python scripts/audit_release.py --arxiv-zip path/to/arxiv_source.zip
python scripts/audit_release.py --arxiv-source path/to/extracted/arxiv_source
```

## Pull Requests

Please keep changes scoped and include tests for:

- permutation grammar changes,
- scoring normalization changes,
- metric aggregation changes,
- manifest generation changes,
- provider environment checks,
- artifact manifest or release-audit changes.

Do not add upstream dataset text, images, prompts, API keys, local caches, or raw provider logs to the repo. Use stable IDs and derived/sanitized artifacts instead.

## New Datasets

Add the dataset to `configs/datasets.yaml`, document license and split information, and expose a loader or trial JSONL adapter. Non-commercial datasets should be clearly marked.
