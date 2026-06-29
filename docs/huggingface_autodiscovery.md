# HuggingFace Auto-Discovery Feasibility

This note evaluates whether Facet-Probe can consume a HuggingFace dataset URL or
dataset ID, inspect metadata and sample rows, and automatically align the dataset
with existing ordering facets.

## Short Answer

Yes, this is feasible as an inspect-and-suggest workflow. It is not reliable as a
fully automatic, no-review benchmark runner for arbitrary HuggingFace datasets.

The implemented inspect-only entry point is:

```bash
facet-probe inspect-hf DATASET_ID --config CONFIG --split SPLIT --sample 20 --emit-spec
```

With the `hf` extra installed and network access available, the command emits:

- a dataset metadata summary,
- candidate facets such as `option_order` or `image_set_order`,
- warnings about missing gold labels, ambiguous schemas, gated data, or
  unsupported modalities,
- a starter `configs/datasets.yaml` fragment.

Use `--output inspection.json` to save the full inspection object and
`--spec-output dataset_fragment.yaml` to save the starter config fragment.
The command imports HuggingFace libraries lazily, so the base package remains
installable without HF dependencies.

## What Can Be Automated

Facet-Probe already has the core abstraction needed for this:

- `AuditItem`: stable `item_id`, `components`, optional `choices`, `gold`, and
  `question_ref`.
- `FacetSpec`: orderable unit plus scoring mode.
- `trial_manifest_rows`: deterministic `K=6` ordering manifest generation once
  normalized items exist.
- `infer_candidate_facets()`: conservative facet suggestions from recursive
  feature names, modalities, and task categories.
- `facet_probe.hf_inspect.build_hf_inspection()`: pure, offline-testable
  metadata summarization and spec generation.
- `facet-probe validate-items`: preflight validation for normalized
  `AuditItem` JSONL.
- `facet-probe make-report`: summary, group, and item-level report artifacts
  from normalized trial JSONL.

An HF inspector can use `huggingface_hub` and `datasets` to collect:

- dataset card data: license, task categories, modalities, configs, tags,
  citation, and gated/private status;
- builder metadata: configs, splits, feature schemas, and dataset size;
- row-shape samples from `datasets.load_dataset(..., streaming=True)`;
- feature types such as `Image`, `Audio`, `Value`, `Sequence`, nested structs,
  and common column names.

Reliable automatic facet suggestions are possible for common schemas:

| Metadata signal | Candidate facet |
|---|---|
| `choices`, `options`, `answerKey`, `answer_index` | `option_order` |
| `context`, `paragraphs`, `evidence`, `evidence_list` | `evidence_chunk_order` |
| `documents`, `retrieved_documents`, `rank`, `score` | `document_rank_order` |
| `images`, `image_list`, `frames`, HF `Image` features | `image_set_order` |
| mixed image/text/table features or `multimodal` task tags | `mixed_modality_order` |

## What Needs Human Review

Dataset cards and feature schemas are not semantic contracts. Human review is
still needed when:

- the gold label is implicit, nested, or stored as free text;
- choices are split across columns or generated from metadata;
- image paths point into archives, private URLs, or external stores;
- retrieved evidence has no stable rank field;
- the dataset has multiple tasks in one split;
- answer normalization is task-specific;
- a new facet requires a new scoring rule;
- the license, gated status, or non-commercial terms limit redistribution or use.

The library can automatically align with existing facets when it can identify
orderable units and a gold-comparison rule. It should not automatically invent a
new facet without a human-defined unit, scoring rule, and same-ordering control.

This matches the paper appendix framing: metadata-driven routing is only the
front door. A dataset is ready for Facet-Probe only after it exposes the
permutation-unit specification for a supported facet and a gold-comparison rule
that makes answers content-invariant under that permutation.

## Implemented Workflow

1. Inspect metadata.
   Run `facet-probe inspect-hf DATASET_ID` with optional `--config`,
   `--split`, `--revision`, `--sample`, `--output`, and `--spec-output`.

2. Record reproducibility metadata.
   Pin the dataset revision, split, config, feature schema, row-shape summary,
   and review warnings. Do not store upstream content in committed artifacts.

3. Review the starter dataset spec.
   The emitted YAML fragment includes candidate facets, license, split, config,
   and warnings. Human review is required before committing it to
   `configs/datasets.yaml`.

4. Normalize dataset rows.
   Write or reuse a loader that maps upstream rows to `AuditItem` JSONL. Common
   MCQ, evidence-list, image-list, and mixed-modality schemas should need only a
   small adapter; unusual schemas need task-specific normalization.

5. Validate before inference.
   Run `facet-probe validate-items normalized.jsonl --facet FACET` to check
   stable IDs, non-empty components, expected `K=6` permutations, and useful
   gold/choice fields.

6. Generate manifests, run models, and report.
   Use `facet-probe make-manifest`, run provider/local model adapters, then use
   `facet-probe make-report trials.jsonl --output-dir reports/run1` for summary,
   group, and item-level evaluation artifacts.

## Feasibility Verdict

High feasibility for automatic metadata inspection and facet recommendation.
Medium feasibility for automatically producing normalized `AuditItem` previews
for common MCQ, image-list, and evidence-list datasets.
Low feasibility for arbitrary no-review adapter generation, because HuggingFace
schemas do not consistently encode gold semantics, retrieval rank semantics, or
modality rendering requirements.

The safest public design is therefore a human-in-the-loop assistant: inspect,
suggest, emit a starter spec/template, validate normalized items, and then reuse
Facet-Probe's deterministic manifest and metric machinery.
