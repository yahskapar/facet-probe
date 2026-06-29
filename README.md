# Facet-Probe

> [!CAUTION]
> This is a very initial (`v0.0.1`) release of the codebase
> corresponding to the release of the
> [pre-print on arXiv](https://arxiv.org/abs/2606.26079). In order to bring
> together various pieces of code and evaluation artifacts, coding agents
> (e.g., Claude Code, OpenAI Codex) were significantly used. Despite high-level
> validation of various scripts and double-checking that they reproduce results
> reported in the pre-print, there still may be issues with the code provided
> as is. As such, please utilize this repo with appropriate caution!

<p align="center">
Official code and artifact release for<br>
<strong>Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models</strong>
</p>

<p align="center">
<a href="https://github.com/yahskapar/facet-probe/actions/workflows/tests.yml">
<img alt="tests" src="https://github.com/yahskapar/facet-probe/actions/workflows/tests.yml/badge.svg">
</a>
</p>

<p align="center">
:fire: Please star this repo if you find it useful, open an issue if you're running into any problems with this repo or have questions, and cite the paper if you reference it or use Facet-Probe in research. :fire:
</p>

<p align="center">
<a href="https://arxiv.org/abs/2606.26079">Pre-print</a> |
<a href="#quickstart">Quickstart</a> |
<a href="#release-checklist">Checklist</a> |
<a href="#release-contents">Release Contents</a> |
<a href="#install">Install</a> |
<a href="#common-usage">Common Usage</a> |
<a href="#python-library-usage">Python API</a> |
<a href="#paper-artifacts">Artifacts</a> |
<a href="#reproducing-the-release">Reproduction</a> |
<a href="#method-notes">Method Notes</a> |
<a href="#extending-facet-probe">Extending</a> |
<a href="#citation">Citation</a> |
<a href="#license">License</a>
</p>

Facet-Probe audits whether multimodal large language models give the same answer
when semantically equivalent evidence is presented in a different order. The
paper studies 18 frontier and open-weight MLLMs over 12 datasets, 5 main
ordering facets, and more than 400,000 trials.

This is the `v0.0.1` initial public release for the arXiv paper. It includes
the cleaned Python package, declarative manifests, release-audit scripts, and
sanitized public evaluation artifacts needed to inspect and reproduce the
reported aggregate results.

The repo does **not** redistribute upstream dataset text, images, tables, raw
provider caches, or API credentials. Dataset-backed reproduction loads from the
original sources under their licenses and terms.

## Quickstart

Install Facet-Probe inside an environment you control:

```bash
# Existing conda environment
conda activate my-env
python -m pip install -e ".[dev,hf,analysis,models,providers]"
python -m pip install -e ".[accelerators]" || \
  echo "Optional accelerator install failed; using portable fallback kernels."

# Provided conda environment
bash setup.sh conda
conda activate facet-probe

# uv environment
bash setup.sh uv
source .venv/bin/activate
```

`setup.sh` defaults to the practical paper-run stack
`dev,hf,analysis,models,providers` and then tries the `accelerators` extra in
best-effort mode. If FlashAttention/Flash Linear Attention/CUDA extension
installation is unsupported on the local machine, setup continues with the
portable PyTorch fallback. Use `--accelerators yes` to require fast kernels, or
`--accelerators no` to skip that optional step.

For full Bayesian ODI/IRT fitting with `facet-probe irt-fit`, include the
heavier `irt` extra in the same install step:

```bash
bash setup.sh uv --extras dev,hf,analysis,models,providers,irt
# or
bash setup.sh conda --extras dev,hf,analysis,models,providers,irt
```

Commands that reference `examples/`, `configs/`, `artifacts/`, or `scripts/`
assume you are running from the repository root. Package-only installs still
include the release configs and compact artifacts, but not the repo examples or
audit script files.

Run the configured paper-profile benchmark for one paper HuggingFace model:

```bash
facet-probe paper-run \
  --model-config configs/models.yaml \
  --models qwen3-5-4b \
  --output-dir runs/qwen3-5-4b-paper
```

This is a real inference run over the configured paper datasets and facets for
the selected paper Qwen3.5-VL model. It does not run every paper model. Use
`--prepare-only` to write the run profile without loading datasets or calling a
model, and omit model selectors only when you intentionally want the full
configured model set.

For a quick development run, restrict the dataset set and item count:

```bash
facet-probe paper-run \
  --model-config configs/models.yaml \
  --models qwen3-5-4b \
  --datasets mmlu_pro \
  --limit-items 2 \
  --output-dir runs/qwen3-5-4b-mmlu-smoke
```

Run one for a supported closed-source provider:

```bash
GOOGLE_API_KEY="..." facet-probe paper-run \
  --provider google \
  --api-model gemini-3.1-pro-preview \
  --api-key-env GOOGLE_API_KEY \
  --output-dir runs/gemini-paper
```

Run a model set from YAML and change the main benchmark parameters:

```bash
facet-probe paper-run \
  --model-config configs/models.yaml \
  --models gemini-3.1-pro-preview qwen3-5-4b \
  --k 6 \
  --seed 42 \
  --output-dir runs/paper-config
```

Prepare the full configured paper model/dataset profile without launching it:

```bash
facet-probe paper-run --output-dir runs/full-paper --prepare-only
```

Each executed run directory contains `run_profile.json`, `provider_status.json`,
`models.jsonl`, `datasets.jsonl`, `manifest.jsonl`, `trials.jsonl`,
`summary.json`, `group_summary.csv`, `run_status.json`, and `report/`.
Long-running commands print timestamped progress/status messages to stderr by
default, for example `facet-probe paper-run [12:34:56] loading runtime examples`.
Final JSON/table payloads remain on stdout; use `--quiet` to suppress
Facet-Probe status messages. These files give you the direct run-level audit
results: flip rate, OSI, macro accuracy when gold labels are available, item
metrics, and grouped summaries. The command does not silently run a Bayesian
ODI/IRT posterior fit.
Use `--prepare-only` when you only want the reproducible run profile and do not
want to load datasets or call a model yet. Full paper-profile runs are strict by
default: if a dataset/facet cannot reach its configured audited item count, the
command fails instead of reporting a partial run as complete. Use
`--allow-partial` only for development/debugging. Paper-profile execution uses
dataset-specific loaders for the configured paper datasets; arbitrary added
HuggingFace datasets use the generic adapter templates unless you add a custom
loader. HuggingFace datasets stream by default where supported; use
`--no-streaming` when you intentionally want normal cached dataset loading.
Facet-Probe retries transient HuggingFace dataset/API failures, and the MMLU-Pro
paper loader falls back to the public test parquet URL if HuggingFace's dataset
tree endpoint times out. File/archive-backed public assets for mixed-modality
datasets may still require large upstream downloads.
Local HuggingFace adapters use `fast_mode: auto` by default: they try
accelerated attention/kernel paths first and retry with portable PyTorch/SDPA
settings if the fast path is unavailable. Set `FACET_PROBE_HF_FAST_MODE=off` to
skip fast-path attempts, or `FACET_PROBE_HF_FAST_MODE=require` to fail instead
of falling back.

Run generation and the paper-style mixed-modality semantic judge in one command:

```bash
GOOGLE_API_KEY="..." facet-probe paper-run \
  --model-config configs/models.yaml \
  --models qwen3-5-4b \
  --output-dir runs/qwen3-5-4b-paper \
  --judge-mixed
```

Judge mixed-modality free-form outputs from a completed run:

```bash
GOOGLE_API_KEY="..." facet-probe judge-mixed runs/qwen3-5-4b-paper/trials.jsonl \
  --judge mixed-semantic-primary \
  --output-dir runs/qwen3-5-4b-paper/mixed_semantic_judge
```

This writes `mixed_semantic_judgments.jsonl`,
`mixed_semantic_summary.json`, and `mixed_semantic_summary.csv`. The default
judge profile is `mixed-semantic-primary` from `configs/models.yaml`, which uses
Gemini-Pro with `GOOGLE_API_KEY`. The input trial file must contain
`mixed_modality_order` rows. To run a cross-vendor judge instead, use
`--judge mixed-semantic-cross-vendor-openai` with `OPENAI_API_KEY`, or override
the config with `--provider`, `--api-model`, and `--api-key-env`.

Inspect the released paper ODI/IRT artifacts from the installed package:

```bash
facet-probe irt-summary --output-dir reports/released_irt
```

This writes a compact summary bundle, theta CSVs, diagnostics, and copies of the
released ODI artifacts, including the Table 2 modal-outcome facet decomposition
and appendix posterior intervals.

Fit the public Bayesian ODI/IRT model directly from a completed run:

```bash
facet-probe irt-fit runs/qwen3-5-4b-paper/trials.jsonl \
  --outcome modal \
  --output-dir runs/qwen3-5-4b-paper/irt_fit_modal
```

When given raw run `trials.jsonl`, `irt-fit` first writes the deterministic
modal/correct outcome export under `runs/qwen3-5-4b-paper/irt_fit_modal/irt_input/`,
then fits the model. Use `--dry-run` to validate and summarize those fit inputs
without importing PyMC or sampling.

You can also materialize the export as a separate inspectable/reusable step:

```bash
facet-probe irt-export runs/qwen3-5-4b-paper/trials.jsonl \
  --output-dir runs/qwen3-5-4b-paper/irt_input
```

This creates `irt_input_trials.csv`, `irt_input_trials.jsonl`,
`irt_input_groups.csv`, and `irt_input_summary.json`; the same CSV/JSONL can be
passed to `irt-fit` if you want to avoid regenerating inputs across repeated
fits.

For a paper-scale run, install the `irt` extra and expect this to be a heavier
posterior-sampling job:

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
theta summaries, diagnostics, and `irt_fit_summary.json`. PyMC sampling progress
is shown by default during real fits; use `--no-progressbar` to suppress sampler
progress and `--quiet` to suppress Facet-Probe status messages. Add
`--save-idata` only when you also want the full ArviZ NetCDF trace. The public
fit implementation may be further optimized in future releases.

Use the same profile from Python:

```python
import facet_probe as fp

profile = fp.paper_profile(
    config_dir="configs",
    models=["gemini-3.1-pro-preview", "qwen3-5-4b"],
)
judge = fp.judge_profile(config_dir="configs")
print(profile.k_orderings, profile.seed, len(profile.datasets), judge.name)
```

Add a new HuggingFace dataset:

```bash
facet-probe inspect-hf allenai/ai2_arc --config ARC-Challenge --split validation --emit-spec
```

```python
new_dataset = fp.hf_dataset(
    "allenai/ai2_arc",
    name="arc_challenge",
    config="ARC-Challenge",
    split="validation",
    facets=["option_order"],
)
custom_only = profile.only_datasets(new_dataset, name="arc-challenge-only")
paper_plus_custom = profile.add_datasets(new_dataset, name="paper-plus-arc-challenge")
```

Create a manifest for model adapters:

```python
row = {
    "id": "demo-001",
    "question": "Which option is the target color?",
    "choices": ["red", "blue", "green", "yellow"],
    "answer": "2",
}
item = fp.mcq_audit_item(row, dataset=new_dataset.name)
fp.validate_audit_items([item], facet="option_order", k=custom_only.k_orderings)

manifest = fp.trial_manifest_rows(
    [item],
    facet="option_order",
    k=custom_only.k_orderings,
    seed=custom_only.seed,
    include_ordered_components=True,
)

prompt = fp.render_ordered_text_prompt(
    item,
    manifest[0]["ordered_component_ids"],
    question=row["question"],
    resolve_content=lambda component: component.content_ref,
)

print(custom_only.name, paper_plus_custom.name, len(manifest))
```

See the same objects printed without downloading models or calling APIs:

```bash
python examples/quickstart_profile.py
```

## Release Checklist

This checklist tracks the public release state and follow-on milestones.

- [x] `v0.0.1` initial public code and evaluation artifacts release.
- [ ] Further validation and testing of `v0.0.1` release
- [ ] Expand dataset-adapter templates and add more model-adapter examples.
- [ ] Publicly release expanded evaluation artifacts.
- [ ] Update and release Facet-Probe `v1.0.0`.

## Release Contents

- `src/facet_probe/`: permutation, scoring, metrics, manifest, runner, semantic judging, provider-env, artifact, and dataset-registry code.
- `configs/`: dataset, model, facet, ordering, and release-artifact manifests.
- `artifacts/`: sanitized aggregate tables, compact ODI outputs, robustness, mitigation, screens, and provenance notes.
- Packaged wheels include a read-only copy of `configs/` and `artifacts/` so
  installed CLI/API calls can still load the public release profile and verify
  compact artifacts outside a repo checkout.
- `scripts/audit_release.py`: offline release audit for manifest coverage, artifact consistency, sanitization, and secret scans.
- `examples/toy_items.jsonl`: minimal canonical item file for trying bulk manifest generation.
- `examples/python_library_usage.py`: minimal import-based Python API example.
- `examples/quickstart_profile.py`: validated profile-driven paper/custom dataset quickstart.
- `tests/`: unit tests for the public contracts.

## Install

Facet-Probe supports both `conda` and `uv` setup paths. Both default to
Python 3.11.

Using `conda`:

```bash
bash setup.sh conda
conda activate facet-probe
```

If your conda env directory is not writable, use a prefix:

```bash
bash setup.sh conda --prefix /path/to/envs/facet-probe
conda activate /path/to/envs/facet-probe
```

Using `uv`:

```bash
bash setup.sh uv
source .venv/bin/activate
```

If you use the `uv` setup path, run it outside an active conda environment
unless you explicitly pass `--allow-active-conda`. uv will use or provision
Python 3.11 by default; pass `--python 3.12` or another supported version only
when you want to override that default.

Install additional optional extras with either setup path:

```bash
bash setup.sh uv --extras dev,hf,analysis
bash setup.sh conda --extras dev,hf,analysis
bash setup.sh uv --extras dev,hf,analysis,irt
bash setup.sh conda --extras dev,hf,analysis,irt
```

Use `dev,hf,analysis,irt` only if you plan to refit the Bayesian ODI model
locally with `facet-probe irt-fit`.

Manual pip fallback:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Install directly from a public Git tag:

```bash
python -m pip install "facet-probe @ git+https://github.com/yahskapar/facet-probe.git@v0.0.1"
uv pip install "facet-probe @ git+https://github.com/yahskapar/facet-probe.git@v0.0.1"
```

For conda users, activate the conda environment first and then run the
`python -m pip install ...` command inside it.

When maintainers publish a later package-index release, installation becomes:

```bash
python -m pip install facet-probe
uv pip install facet-probe
```

The wheel includes the release configs and compact artifacts, so
`facet-probe verify-artifacts` and `facet_probe.paper_profile()` work even when
called outside a cloned repository. Full reruns still download upstream datasets
and call local/API models at runtime.

That PyPI step is not required for the `v0.0.1` GitHub release. Maintainers can
publish a package-index release by building wheel/sdist artifacts from this repo
and uploading them to PyPI, preferably first to TestPyPI:

```bash
python -m build
python -m twine upload dist/*
```

`uv build` is also suitable for building the local package artifacts. A conda
package is a separate conda-forge-style release path and is not required for
people to install Facet-Probe inside conda environments.

## Common Usage

List the paper facets and datasets:

```bash
facet-probe list-facets
facet-probe list-datasets
```

Generate deterministic orderings for one item:

```bash
facet-probe make-permutations --n-components 5 --item-id mmlu_pro::example --k 6
```

Build a bulk trial manifest from canonical item JSONL:

```bash
facet-probe make-manifest examples/toy_items.jsonl --facet option_order --output /tmp/toy_manifest.jsonl
```

Validate normalized items before launching model calls:

```bash
facet-probe validate-items examples/toy_items.jsonl --facet option_order
```

Inspect a HuggingFace dataset and emit a starter dataset spec:

```bash
facet-probe inspect-hf TIGER-Lab/MMLU-Pro --split test --sample 20 --emit-spec
```

This command requires the `hf` extra and network access.

Check provider environment variables without printing secret values:

```bash
facet-probe check-env --providers google openai anthropic
```

The command exits non-zero when a required provider variable is missing.

Verify the included aggregate artifacts:

```bash
facet-probe verify-artifacts
```

Audit normalized trial JSONL:

```bash
facet-probe audit-jsonl path/to/trials.jsonl --group-csv group_summary.csv
facet-probe make-report path/to/trials.jsonl --output-dir reports/run1
facet-probe make-report examples/toy_trials.jsonl --output-dir /tmp/toy_report
```

Trial JSONL rows should contain fields such as `facet`, `dataset`, `model`,
`item_id`, `ordering_idx`, `answer_normalized`, and `correct`.

## Python Library Usage

Use Facet-Probe directly from Python when building dataset or model adapters:

```python
import facet_probe as fp

items = [
    fp.mcq_audit_item(
        {
            "id": "001",
            "question": "Which option is the target color?",
            "choices": ["red", "blue", "green", "yellow"],
            "answer": "2",
        },
        dataset="toy_mcq",
    )
]

fp.validate_audit_items(items, facet="option_order")
manifest = fp.trial_manifest_rows(items, facet="option_order", k=6, seed=42)
```

Run the complete minimal example:

```bash
python examples/python_library_usage.py
```

## Paper Artifacts

High-signal release artifacts live under `artifacts/`:

- `artifacts/paper/per_facet_per_model.csv`: Table 1 values, including screened image-set and mixed-modality sem-flip.
- `artifacts/paper/panel_means.csv`: panel means for the 5 main facets.
- `artifacts/paper/capability_summary.csv`: Figure 1 capability/scaling summary values.
- `artifacts/paper/additional_facet_results.csv`: appendix demoted-facet and tool-description null/stress summary values.
- `artifacts/odi/facet_decomposition.csv`: Table 2 screened modal-outcome ODI facet decomposition.
- `artifacts/odi/posterior_intervals.csv`: appendix ODI posterior interval summaries.
- `artifacts/odi/*theta.json`: modal/correct model ability summaries used by IRT analyses.
- `artifacts/odi/*per_item_params.parquet`: compact per-item posterior parameter summaries.
- `artifacts/diagnostics/calibration_mechanism_summary.csv`: compact Q6 calibration and mechanism-classification values.
- `artifacts/diagnostics/llm_judge_validation.csv`: compact LLM-judge validation values.
- `artifacts/robustness/decoder_decomp_screened_by_facet.csv`: screened decoder-noise vs ordering decomposition.
- `artifacts/mitigation/policy_means.csv`: Table 4 screened mitigation cost-Pareto policy means.
- `artifacts/mitigation/cta_flip_summary.csv`: Figure 4 CTA baseline-vs-intervention cells.
- `artifacts/mitigation/think_budget_sweep.csv`: Figure 4 hard/easy think-budget sweep cells.
- `artifacts/screens/imageset_position_reference_screen.csv`: sanitized image-set screen IDs and labels.

`configs/release_artifacts.yaml` lists every shipped artifact and maps each one
to the paper table, figure, appendix section, or provenance note it supports.
`artifacts/reproducibility_matrix.md` gives the same mapping in a readable
table. See `docs/artifacts.md` for the release boundary and intentional
exclusions.

## Reproducing The Release

For a fresh clone, the lightweight reproducibility gate is:

```bash
facet-probe verify-artifacts
python scripts/audit_release.py --offline-only
python -m pytest
```

To cross-check the release against the public arXiv source bundle:

```bash
python scripts/audit_release.py --arxiv-zip path/to/arxiv_source.zip
python scripts/audit_release.py --arxiv-source path/to/extracted/arxiv_source
```

The public release is self-contained for the shipped artifact claims: it
includes the compact tables, screen summaries, ODI summaries, provenance notes,
configs, and verification code needed to inspect the arXiv aggregate results.
Full raw reruns require upstream datasets and model/API access under their
original terms, so raw provider outputs and upstream content are listed as
future expanded artifacts rather than redistributed in `v0.0.1`.

The release audit checks that all shipped artifacts are manifest-listed, compact
numeric artifacts match expected paper values, sanitized screens omit upstream
dataset content, setup commands are documented, and no obvious credentials
appear in text files.

Provider credentials are read only from environment variables. Start from
`.env.example` and never commit a filled `.env` file.

## Method Notes

Facet-Probe uses `K=6` orderings per item by default. When `n! >= 6`, it
samples without replacement and includes canonical ordering first; when
`n! < 6`, it cycles all unique permutations. Public permutation generation uses
a SHA-256-derived stable seed, not Python's process-randomized `hash()`.

For `option_order`, the selected displayed letter is mapped back through the
inverse permutation to the source option content index. Flips therefore measure
changes in selected option content, not trivial changes in letter labels.

For `image_set_order`, clean paper summaries apply a position-reference screen:
items whose true gold answer moves when images are permuted are excluded from
clean image-set estimates.

For `mixed_modality_order`, the paper artifacts report the structured
LLM-judge/semantic-flip summaries described in the paper and
`docs/artifacts.md`. The public runner can generate new mixed-modality
judgments with `facet-probe judge-mixed` or `paper-run --judge-mixed`. The
historical full judged raw-output release remains a planned expanded artifact.

For ODI/IRT, the main paper decomposition uses the modal outcome: a trial is 1
when its normalized answer matches that model/item's untied modal answer across
orderings. Correct-outcome theta summaries are included for ability and
capability analyses. `facet-probe irt-summary` exposes the released paper
outputs, and `facet-probe irt-fit` accepts either a new run `trials.jsonl` file
or modal/correct outcome rows previously written by `facet-probe irt-export`.
The export command remains useful when you want to inspect, share, or reuse the
deterministic fit input separately from the heavier Bayesian fitting step.

## Extending Facet-Probe

New datasets can be added through `configs/datasets.yaml`, the dataset metadata
helpers in `src/facet_probe/datasets.py`, and, when needed, a loader or template
that maps upstream rows to `AuditItem` objects. The minimum contract is:

- a list of orderable units for the target facet,
- a deterministic item ID,
- a gold-comparison rule or judge label,
- a permutation manifest generated with the same `K` and seed.

`facet-probe inspect-hf` and `facet_probe.hf_inspect.build_hf_inspection()`
provide conservative facet suggestions from HuggingFace dataset metadata,
recursive feature schemas, and sample row shapes. `facet-probe validate-items`
checks normalized `AuditItem` JSONL before inference, and
`facet-probe make-manifest` turns those items into trial manifests for
provider-specific or local-model adapters. `facet_probe.templates` provides
starter adapters for common MCQ, evidence-list, image-list, and mixed-modality
row shapes. See
`docs/huggingface_autodiscovery.md` for the feasibility and limits of consuming
arbitrary HuggingFace dataset URLs, and `docs/adapter_templates.md` for
template examples.

Closed-source model adapters should use environment variables such as
`GOOGLE_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY`; open-weight
adapters should record the HuggingFace repo, dtype, quantization, generation
kwargs, and access date.

## Citation

`CITATION.cff` is repo metadata for GitHub and citation managers; its
preferred citation is the same paper citation shown here.

```bibtex
@article{paruchuri2026same,
  title={Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models},
  author={Paruchuri, Akshay and Koyejo, Sanmi and Adeli, Ehsan},
  journal={arXiv preprint arXiv:2606.26079},
  year={2026}
}
```

## License

Code, configs, docs, and repo-native derived artifacts are released under
Apache-2.0 unless otherwise noted. Upstream datasets, model outputs, provider
APIs, and third-party dependencies remain governed by their own licenses and
terms. See `ARTIFACT_LICENSES.md`.
