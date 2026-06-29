import sys
from types import SimpleNamespace

from facet_probe.hf_inspect import (
    build_hf_inspection,
    flatten_feature_schema,
    inspect_hf_dataset,
    normalize_hf_dataset_id,
)


def test_normalize_hf_dataset_id_accepts_dataset_url():
    dataset_id = normalize_hf_dataset_id("https://huggingface.co/datasets/org/my-dataset/tree/main")

    assert dataset_id == "org/my-dataset"


def test_flatten_feature_schema_handles_nested_mappings():
    features = {
        "question": "string",
        "choices": {"text": ["string"]},
        "documents": [{"title": "string", "rank": "int32"}],
    }

    paths = {item.path for item in flatten_feature_schema(features)}

    assert "choices.text[]" in paths
    assert "documents[].rank" in paths


def test_build_hf_inspection_suggests_facets_and_starter_spec_without_content():
    inspection = build_hf_inspection(
        dataset_id="org/visual-rag",
        split="validation",
        license="Apache-2.0",
        task_categories=("visual-question-answering",),
        features={
            "question": "string",
            "images": ["Image"],
            "evidence": [{"text": "string", "rank": "int32"}],
            "answer": "string",
        },
        sample_rows=[
            {
                "question": "not stored in the summary",
                "images": ["image-ref"],
                "evidence": [{"text": "also not stored", "rank": 1}],
                "answer": "A",
            }
        ],
    )

    assert inspection.dataset_id == "org/visual-rag"
    assert "image_set_order" in inspection.candidate_facets
    assert "document_rank_order" in inspection.candidate_facets
    assert "mixed_modality_order" in inspection.candidate_facets
    assert inspection.starter_dataset_spec["visual_rag"]["hf_repo"] == "org/visual-rag"
    assert inspection.sample_profiles[0].observed_types


def test_build_hf_inspection_uses_public_missing_split_label():
    inspection = build_hf_inspection(
        dataset_id="org/no-split",
        features={"question": "string", "choices": ["string"], "answer": "string"},
    )

    assert inspection.starter_dataset_spec["no_split"]["split"] == "unspecified"


def test_inspect_hf_dataset_reports_progress(monkeypatch):
    class FakeApi:
        def dataset_info(self, dataset_id, revision=None):
            assert dataset_id == "org/demo"
            assert revision is None
            return SimpleNamespace(
                tags=["task_categories:question-answering"],
                cardData={"license": "mit", "task_categories": ["question-answering"]},
            )

    fake_datasets = SimpleNamespace(
        get_dataset_config_names=lambda *_args, **_kwargs: ["default"],
        get_dataset_split_names=lambda *_args, **_kwargs: ["validation"],
        load_dataset=lambda *_args, **_kwargs: iter(
            [{"question": "Q?", "choices": ["A", "B"], "answer": "A"}]
        ),
        load_dataset_builder=lambda *_args, **_kwargs: SimpleNamespace(
            info=SimpleNamespace(
                features={"question": "string", "choices": ["string"], "answer": "string"},
                splits={"validation": object()},
                license="mit",
            )
        ),
    )
    fake_hub = SimpleNamespace(HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    messages = []

    inspection = inspect_hf_dataset(
        "org/demo",
        sample=1,
        progress_callback=messages.append,
    )

    assert inspection.dataset_id == "org/demo"
    assert inspection.sample_row_count == 1
    assert "loading dataset card metadata" in messages
    assert "loading up to 1 sample row(s) from split=validation" in messages
    assert "loaded 1 sample row(s)" in messages
