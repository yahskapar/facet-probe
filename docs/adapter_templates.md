# Adapter Templates

Facet-Probe ships small adapter-template helpers for common dataset row shapes.
They emit canonical `AuditItem` objects with stable content references; upstream
text, images, and tables should stay in the runtime loader or provider adapter.

## Dataset Rows

Multiple-choice rows:

```python
import facet_probe as fp

item = fp.mcq_audit_item(
    {
        "id": "row-1",
        "question": "Which option is correct?",
        "choices": {"label": ["A", "B"], "text": ["red", "blue"]},
        "answer": "1",
    },
    dataset="my_mcq_dataset",
)
```

Evidence-list rows:

```python
item = fp.evidence_list_audit_item(
    {
        "id": "row-2",
        "question": "What does the evidence support?",
        "evidence": [{"text": "first passage"}, {"text": "second passage"}],
        "answer": "target",
    },
    dataset="my_evidence_dataset",
)
```

Image-list and mixed-modality rows:

```python
image_item = fp.image_list_audit_item(
    {"id": "row-3", "images": ["image-a", "image-b"], "answer": "A"},
    dataset="my_image_dataset",
)

mixed_item = fp.mixed_modality_audit_item(
    {"id": "row-4", "captions": ["caption"], "images": ["image"]},
    dataset="my_mixed_dataset",
    component_fields=(("captions", "text"), ("images", "image")),
)
```

## Model Adapter Rendering

After `facet-probe make-manifest` or `fp.trial_manifest_rows`, provider-specific
adapters can render one ordered prompt by resolving component references at
runtime:

```python
prompt = fp.render_ordered_text_prompt(
    item,
    manifest_row["ordered_component_ids"],
    question="Which option is correct?",
    resolve_content=lambda component: runtime_store[component.content_ref],
)
```

Closed-source adapters should call the provider with fixed model ID, generation
settings, access date, and environment-backed credentials. Open-weight adapters
should record the HuggingFace repo, dtype, quantization, generation kwargs, and
access date. Both should write normalized trial JSONL records with `facet`,
`dataset`, `model`, `item_id`, `ordering_idx`, `permutation`,
`answer_normalized`, `gold_normalized`, `correct`, and `score_kind`.
