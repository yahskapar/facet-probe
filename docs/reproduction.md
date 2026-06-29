# Reproduction

## Verify Shipped Artifacts

Set up either a conda or uv environment. Both paths default to Python 3.11:

```bash
bash setup.sh conda
conda activate facet-probe
```

Use `bash setup.sh conda --prefix /path/to/envs/facet-probe` if your default
conda environment directory is not writable.

or:

```bash
bash setup.sh uv
source .venv/bin/activate
```

Then run:

```bash
facet-probe verify-artifacts
python -m pytest
```

This checks panel means and key ODI ratios against the shipped CSV/JSON artifacts.

For public clones without the private paper workspace, run the offline subset:

```bash
python scripts/audit_release.py --offline-only
```

When the full `EMNLP_2026` paper workspace and latest arXiv source ZIP are
available, run the complete source-grounded audit:

```bash
python scripts/audit_release.py \
  --source-root /path/to/EMNLP_2026 \
  --arxiv-zip /path/to/arxiv_latest_post_finalization/paper.zip
```

This verifies artifact manifest coverage, byte-for-byte regeneration from the
paper workspace, arXiv source-section grounding, arXiv table/figure-to-artifact
numeric alignment, sanitized screen boundaries, and absence of obvious
credentials in text files.

Use `artifacts/reproducibility_matrix.md` to trace each paper table, figure, or
appendix result family to the public artifact files and verification command.

## Recreate Release Artifacts From the Paper Workspace

If you have the full `EMNLP_2026` workspace next to this repo:

```bash
python scripts/materialize_release_artifacts.py --source-root ../
```

Use `--source-root /path/to/EMNLP_2026` when the paper workspace is elsewhere.
Use `--output-root /tmp/facet-probe-check` to materialize into a temporary
directory for comparison without touching the working tree.

## Generate Trial Manifests

Normalize dataset rows to `AuditItem` JSONL, then generate `K=6` ordering rows:

```bash
facet-probe make-manifest examples/toy_items.jsonl \
  --facet option_order \
  --output /tmp/toy_manifest.jsonl
```

Each manifest row records `facet`, `dataset`, `item_id`, `ordering_idx`,
`permutation`, `component_ids`, and `ordered_component_ids`. Provider-specific
adapters should render the ordered components, call the model, and write
normalized trial JSONL.

## Provider Credentials

Provider credentials are never stored in configs or artifacts. Use environment
variables and check readiness without printing values:

```bash
facet-probe check-env --providers google openai anthropic
```

See `.env.example` for variable names. Keep real keys in an untracked `.env` or
your shell environment.

## Recompute Metrics From Trial JSONL

Trial JSONL records should include:

```json
{
  "facet": "option_order",
  "dataset": "mmlu_pro",
  "model": "example-model",
  "item_id": "mmlu_pro::123",
  "ordering_idx": 0,
  "permutation": [0, 1, 2, 3],
  "answer_normalized": "2",
  "gold_normalized": "2",
  "correct": true
}
```

Then run:

```bash
facet-probe audit-jsonl trials.jsonl --group-csv group_summary.csv
```

For `option_order`, normalize model letters to source option indices before writing `answer_normalized`, or use `facet_probe.scoring.normalize_answer("option_content_idx", raw, ...)`.

## Full Paper Reproduction

Full reproduction requires:

- upstream datasets listed in `configs/datasets.yaml`,
- model/API access listed in `configs/models.yaml`,
- fixed seed 42,
- `K=6` orderings from `configs/orderings.yaml`,
- image-set screen from `artifacts/screens/`,
- LLM-judge configuration for mixed-modality scoring.

Closed-source provider drift is a limitation. The paper pins the access window to May 4-25, 2026.
