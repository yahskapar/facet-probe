# Extending Facet-Probe

Facet-Probe separates four contracts:

1. Dataset loader: exposes stable item IDs, gold labels, and orderable units.
2. Facet grammar: samples `K=6` permutations over the correct unit.
3. Manifest builder: records the sampled orderings for reproducible inference.
4. Scoring rule: normalizes outputs to a content-invariant answer.

## Add a Dataset

For a HuggingFace dataset, first inspect metadata and feature shapes:

```bash
facet-probe inspect-hf DATASET_ID --split validation --sample 20 --emit-spec
```

Review the emitted starter spec, then add the final version to
`configs/datasets.yaml`. Implement or register a loader that emits canonical
`AuditItem` JSONL:

- `dataset`
- `item_id`
- `components`
- `question_ref`
- `choices`
- `gold`

For HuggingFace datasets, `facet-probe inspect-hf` and
`facet_probe.datasets.infer_candidate_facets()` can suggest a starting facet
set from dataset-card metadata, recursive feature names, declared modalities,
task categories, and sample row shapes. Treat the suggestion as a review aid,
not as proof that the dataset is ready to run.

Use the adapter templates in `facet_probe.templates` for common MCQ,
evidence-list, image-list, and mixed-modality row shapes. See
`docs/adapter_templates.md` for examples.

Validate the normalized items before inference:

```bash
facet-probe validate-items path/to/items.jsonl --facet option_order
```

Then build trial rows:

```bash
facet-probe make-manifest path/to/items.jsonl --facet option_order --output manifest.jsonl
```

After running a model adapter and writing normalized trial JSONL, create report
artifacts:

```bash
facet-probe make-report trials.jsonl --output-dir reports/my_run
```

## Add a Facet

A new facet should define:

- the orderable unit,
- the permutation grammar,
- the output normalization rule,
- a same-ordering control plan.

Register it in `src/facet_probe/facets.py` and `configs/facets.yaml`.

## Add a Model

Model adapters are intentionally thin in this release. A model adapter needs to:

- render the ordered components in provider-specific format,
- call the model with fixed generation settings,
- preserve raw output for audit,
- write normalized trial JSONL records.

Closed-source adapters should record provider model IDs and access dates.
Open-weight adapters should record the HuggingFace repo, dtype, quantization,
and generation kwargs.

Use `facet-probe check-env --providers ...` before closed-source runs. The
checker reports only whether required variables are set; it never prints key
values.
