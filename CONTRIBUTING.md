# Contributing

Thanks for helping improve Facet-Probe.

## Development Setup

```bash
bash setup.sh uv
source .venv/bin/activate
pytest
facet-probe verify-artifacts
```

Conda users can instead run:

```bash
bash setup.sh conda
conda activate facet-probe
```

When the local paper workspace is available, also run:

```bash
python scripts/audit_release.py --source-root /path/to/EMNLP_2026 --arxiv-zip /path/to/paper.zip
```

## Pull Requests

Please keep changes scoped and include tests for:

- permutation grammar changes,
- scoring normalization changes,
- metric aggregation changes,
- manifest generation changes,
- provider environment checks,
- artifact materialization changes.

Do not add upstream dataset text, images, prompts, API keys, local caches, or raw provider logs to the repo. Use stable IDs and derived/sanitized artifacts instead.

## New Datasets

Add the dataset to `configs/datasets.yaml`, document license and split information, and expose a loader or trial JSONL adapter. Non-commercial datasets should be clearly marked.
