# Facet-Probe

<p align="center">
Official code and artifact release for<br>
<strong>Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models</strong>
</p>

<p align="center">
Please star this repo if you find it useful, open an issue if something is unclear, and cite the paper if you use Facet-Probe in research.
</p>

<p align="center">
<a href="#install">Install</a> |
<a href="#quick-start">Quick Start</a> |
<a href="#paper-artifacts">Artifacts</a> |
<a href="#reproducing-the-release">Reproduction</a> |
<a href="#release-checklist">Checklist</a> |
<a href="#extending-facet-probe">Extending</a>
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

## Release Contents

- `src/facet_probe/`: permutation, scoring, metrics, manifest, provider-env, artifact, and dataset-registry code.
- `configs/`: dataset, model, facet, ordering, and release-artifact manifests.
- `artifacts/`: sanitized aggregate tables, compact ODI outputs, robustness, mitigation, screens, and provenance notes.
- `scripts/materialize_release_artifacts.py`: deterministic materializer from the private paper workspace.
- `scripts/audit_release.py`: release audit for manifest coverage, artifact regeneration, arXiv-source grounding, sanitization, and secret scans.
- `examples/toy_items.jsonl`: minimal canonical item file for trying bulk manifest generation.
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
```

Use `dev,hf,analysis,irt` only if you plan to refit the Bayesian ODI model
locally.

Manual pip fallback:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

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

Check provider environment variables without printing secret values:

```bash
facet-probe check-env --providers google openai anthropic
```

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

## Paper Artifacts

High-signal release artifacts live under `artifacts/`:

- `artifacts/paper/per_facet_per_model.csv`: Table 1 values, including screened image-set and mixed-modality sem-flip.
- `artifacts/paper/panel_means.csv`: panel means for the 5 main facets.
- `artifacts/odi/facet_decomposition.csv`: Table 2 screened modal-outcome ODI facet decomposition.
- `artifacts/odi/posterior_intervals.csv`: appendix ODI posterior interval summaries.
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

For authors or reviewers with the full local `EMNLP_2026` paper workspace:

```bash
python scripts/materialize_release_artifacts.py --source-root ../
python scripts/audit_release.py --source-root ../ \
  --arxiv-zip ../arxiv_latest_post_finalization/_EMNLP_2026__Same_Evidence__Different_Answer__Auditing_Order_Sensitivity_in_Multimodal_Large_Language_Models.zip
```

The release audit checks that all shipped artifacts are manifest-listed, all
materialized artifacts regenerate byte-for-byte from the paper workspace when
`--source-root` is supplied, the latest arXiv source contains the expected
quantitative sections when `--arxiv-zip` is supplied, sanitized screens omit
upstream dataset content, and no obvious credentials appear in text files.

Provider credentials are read only from environment variables. Start from
`.env.example` and never commit a filled `.env` file.

## Release Checklist

This checklist tracks the public release state and the most useful follow-on
milestones.

- [x] `v0.0.1` initial public code and evaluation artifacts release.
  Includes the cleaned Python package, Apache-2.0 licensing, conda and uv setup
  paths, sanitized paper artifacts, provenance notes, release audits, Figure 4
  CTA/think-budget summaries, HuggingFace metadata inspection, normalized item
  validation, and evaluation-report helpers. Verified locally with
  `python -m pytest`, `ruff check .`, `facet-probe verify-artifacts`, offline
  and source-grounded release audits, setup dry-runs, package-build smoke test,
  and placeholder/secret scans.
- [ ] Add dataset adapter templates for common dataset types and more
  model-adapter examples.
  Prioritize MCQ, evidence-list, image-list, and mixed-modality dataset
  templates, plus one closed-source provider example and one local/open-weight
  HuggingFace model example.
- [ ] Publicly release expanded evaluation artifacts.
  Candidate artifacts include full permutation manifests, normalized trial
  outputs, and other higher-granularity evaluation files after provider,
  license, storage, and redistribution review.
- [ ] Update and release the full `v1.0.0` version of Facet-Probe.
  This should include the stable public API, expanded adapters/artifacts that
  pass review, public CI or equivalent automated checks, a versioned release,
  and archival citation metadata.

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

For `mixed_modality_order`, free-form outputs are scored through structured
LLM-judge gold-match labels and reported with the caveats in the paper and
`docs/artifacts.md`.

## Extending Facet-Probe

New datasets can be added through `configs/datasets.yaml` and the dataset
registry in `src/facet_probe/datasets.py`. The minimum contract is:

- a list of orderable units for the target facet,
- a deterministic item ID,
- a gold-comparison rule or judge label,
- a permutation manifest generated with the same `K` and seed.

`facet-probe inspect-hf` and `facet_probe.hf_inspect.build_hf_inspection()`
provide conservative facet suggestions from HuggingFace dataset metadata,
recursive feature schemas, and sample row shapes. `facet-probe validate-items`
checks normalized `AuditItem` JSONL before inference, and
`facet-probe make-manifest` turns those items into trial manifests for
provider-specific or local-model adapters. See
`docs/huggingface_autodiscovery.md` for the feasibility and limits of consuming
arbitrary HuggingFace dataset URLs.

Closed-source model adapters should use environment variables such as
`GOOGLE_API_KEY`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY`; open-weight
adapters should record the HuggingFace repo, dtype, quantization, generation
kwargs, and access date.

## Citation

```bibtex
@misc{paruchuri2026same,
  title = {Same Evidence, Different Answer: Auditing Order Sensitivity in Multimodal Large Language Models},
  author = {Paruchuri, Akshay and Koyejo, Sanmi and Adeli, Ehsan},
  year = {2026},
  note = {arXiv preprint. Code and artifacts: https://github.com/yahskapar/facet-probe}
}
```

## License

Code, configs, docs, and repo-native derived artifacts are released under
Apache-2.0 unless otherwise noted. Upstream datasets, model outputs, provider
APIs, and third-party dependencies remain governed by their own licenses and
terms. See `ARTIFACT_LICENSES.md`.
