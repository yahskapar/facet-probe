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

Run the public release audit:

```bash
python scripts/audit_release.py --offline-only
```

To also verify the compact public artifacts against the public arXiv source
bundle, pass the source ZIP or its extracted directory:

```bash
python scripts/audit_release.py --arxiv-zip path/to/arxiv_source.zip
python scripts/audit_release.py --arxiv-source path/to/extracted/arxiv_source
```

This verifies artifact manifest coverage, paper-value consistency for compact
numeric artifacts, sanitized screen boundaries, conda/uv setup documentation,
and absence of obvious credentials in text files.

Use `artifacts/reproducibility_matrix.md` to trace each paper table, figure, or
appendix result family to the public artifact files and verification command.

The `v0.0.1` release is self-contained for the shipped aggregate claims. It
does not redistribute upstream dataset content, raw provider outputs, API
caches, or multi-gigabyte posterior traces. Those objects are omitted because
they are large, provider-sensitive, or governed by upstream dataset licenses;
the public artifacts include the compact values and checks needed to audit the
paper tables and figures without unreleased raw run outputs.

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

For `option_order`, normalize model letters to source option indices before
writing `answer_normalized`, or use
`facet_probe.scoring.normalize_answer("option_content_idx", raw, ...)`.

## Paper-Profile Reruns

Run the configured paper datasets and facets with one local HuggingFace model:

```bash
facet-probe paper-run \
  --hf-model Qwen/Qwen3.5-VL-4B-Instruct \
  --output-dir runs/qwen-paper
```

This command replaces the configured model set with the selected Qwen model. To
inspect the full configured model/dataset profile without launching inference,
run:

```bash
facet-probe paper-run --output-dir runs/full-paper --prepare-only
```

Remove `--prepare-only` only when the required API keys, local-model resources,
dataset licenses, and storage budget are ready.

The run directory contains `run_profile.json`, `provider_status.json`,
`models.jsonl`, `datasets.jsonl`, `manifest.jsonl`, `trials.jsonl`,
`summary.json`, `group_summary.csv`, `run_status.json`, and `report/`. The
command also prints a JSON status summary to the terminal. It is strict by
default: if a configured paper dataset does not load the audited item count, it
fails rather than silently reporting a partial run as complete. Dataset loaders
stream from HuggingFace where supported; file/archive-backed multimodal assets
still download through the upstream distribution mechanism.

For mixed-modality free-form outputs, compute paper-style semantic flip with a
separate judge:

```bash
GOOGLE_API_KEY="..." facet-probe judge-mixed runs/qwen-paper/trials.jsonl \
  --judge mixed-semantic-primary \
  --output-dir runs/qwen-paper/mixed_semantic_judge
```

This writes `mixed_semantic_judgments.jsonl`,
`mixed_semantic_summary.json`, and `mixed_semantic_summary.csv`. The named judge
profile lives in `configs/models.yaml`; override it with `--judge`, or with
`--provider`, `--api-model`, and `--api-key-env`.

Or run the configured judge immediately after generation:

```bash
GOOGLE_API_KEY="..." facet-probe paper-run \
  --hf-model Qwen/Qwen3.5-VL-4B-Instruct \
  --output-dir runs/qwen-paper \
  --judge-mixed
```

Full reproduction requires:

- upstream datasets listed in `configs/datasets.yaml`,
- model/API access listed in `configs/models.yaml`,
- fixed seed 42,
- `K=6` orderings from `configs/orderings.yaml`,
- image-set screen from `artifacts/screens/`,
- LLM-judge configuration for mixed-modality scoring.

Closed-source provider drift is a limitation. The paper pins the access window
to May 4-25, 2026.
