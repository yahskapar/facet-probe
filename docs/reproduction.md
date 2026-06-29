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

The setup script installs the standard paper-run extras and tries local-model
accelerator kernels by default. If those CUDA/Triton extensions are unsupported
on the machine, setup falls back to portable PyTorch kernels. Use
`--accelerators yes` to require the fast kernels, or `--accelerators no` to skip
that optional install step.

Then run:

```bash
facet-probe verify-artifacts
python -m pytest
```

This checks panel means and key ODI ratios against the shipped CSV/JSON artifacts.

The released ODI/IRT summaries can also be materialized from either a checkout
or an installed wheel:

```bash
facet-probe irt-summary --output-dir reports/released_irt
```

This writes `released_irt_summary.json`, theta CSVs, diagnostics, and copies of
the compact `artifacts/odi/` files used for the Table 2 modal-outcome ODI
decomposition, capability theta summaries, and appendix posterior interval
checks.

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

## Fit IRT/ODI

The default run artifacts are flip/OSI/report outputs. To fit the public
Bayesian 2PL ODI/IRT model over a completed run, pass its trial JSONL directly:

```bash
facet-probe irt-fit runs/qwen3-5-4b-paper/trials.jsonl \
  --outcome modal \
  --output-dir runs/qwen3-5-4b-paper/irt_fit_modal
```

`irt-fit` writes the deterministic fit-input export under
`runs/qwen3-5-4b-paper/irt_fit_modal/irt_input/` before fitting. The modal
outcome is 1 when a trial answer matches that model/item's untied modal answer
across orderings; tied modal-answer item groups are omitted from the modal
export. The correct outcome is 1 when a non-missing correctness label is true.

To inspect, share, or reuse those modal/correct Bernoulli outcome rows
separately from fitting, run the export explicitly:

```bash
facet-probe irt-export runs/qwen3-5-4b-paper/trials.jsonl \
  --output-dir runs/qwen3-5-4b-paper/irt_input
```

The export directory contains `irt_input_trials.csv`,
`irt_input_trials.jsonl`, `irt_input_groups.csv`, and
`irt_input_summary.json`; either exported trial file can be passed to
`irt-fit`.

For a paper-scale fit, install the `irt` optional extra and use the paper-style
sampling settings:

```bash
bash setup.sh uv --extras dev,hf,analysis,irt
facet-probe irt-fit runs/qwen3-5-4b-paper/trials.jsonl \
  --outcome modal \
  --n-chains 4 \
  --n-draws 1500 \
  --n-tune 1500 \
  --target-accept 0.95 \
  --nuts-sampler numpyro \
  --output-dir runs/qwen3-5-4b-paper/irt_fit_modal
```

The fit writes compact per-item parameters, per-facet decomposition CSV/JSON,
theta summaries, diagnostics, and `irt_fit_summary.json`. Use `--dry-run` to
validate and summarize inputs before sampling, and add `--save-idata` only when
you need the full ArviZ NetCDF trace. The public fitting workflow may be
optimized further after `v0.0.1`.

## Paper-Profile Reruns

Run the configured paper datasets and facets with one local HuggingFace model:

```bash
facet-probe paper-run \
  --model-config configs/models.yaml \
  --models qwen3-5-4b \
  --output-dir runs/qwen3-5-4b-paper
```

This command selects the paper Qwen3.5-VL 4B profile from `configs/models.yaml`
instead of replacing the paper profile with an arbitrary HuggingFace ID. To
inspect the full configured model/dataset profile without launching inference,
run:

```bash
facet-probe paper-run --output-dir runs/full-paper --prepare-only
```

Remove `--prepare-only` only when the required API keys, local-model resources,
dataset licenses, and storage budget are ready.

The run directory contains `run_profile.json`, `provider_status.json`,
`models.jsonl`, `datasets.jsonl`, `manifest.jsonl`, `trials.jsonl`,
`summary.json`, `group_summary.csv`, `run_status.json`, and `report/`.
Long-running Facet-Probe commands print timestamped progress/status messages to
stderr by default and keep final JSON/table payloads on stdout. Use `--quiet` to
suppress progress output. It is strict by default: if a configured paper dataset
does not load the audited item count, it fails rather than silently reporting a
partial run as complete.
Dataset loaders stream from HuggingFace where supported, retry transient
HuggingFace API failures, and use a direct public parquet fallback for MMLU-Pro
if the HuggingFace dataset tree endpoint times out. File/archive-backed
multimodal assets still download through the upstream distribution mechanism.
Local HuggingFace model profiles default to `fast_mode: auto`: accelerated
attention/kernel paths are attempted first, then the adapter retries with
portable SDPA/default HuggingFace settings if the fast path cannot load. Set
`FACET_PROBE_HF_FAST_MODE=off` to skip fast attempts, or
`FACET_PROBE_HF_FAST_MODE=require` to fail instead of falling back.

For mixed-modality free-form outputs, compute paper-style semantic flip with a
separate judge:

```bash
GOOGLE_API_KEY="..." facet-probe judge-mixed runs/qwen3-5-4b-paper/trials.jsonl \
  --judge mixed-semantic-primary \
  --output-dir runs/qwen3-5-4b-paper/mixed_semantic_judge
```

This writes `mixed_semantic_judgments.jsonl`,
`mixed_semantic_summary.json`, and `mixed_semantic_summary.csv`. The named judge
profile lives in `configs/models.yaml`; override it with `--judge`, or with
`--provider`, `--api-model`, and `--api-key-env`.

Or run the configured judge immediately after generation:

```bash
GOOGLE_API_KEY="..." facet-probe paper-run \
  --model-config configs/models.yaml \
  --models qwen3-5-4b \
  --output-dir runs/qwen3-5-4b-paper \
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
