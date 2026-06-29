from facet_probe.datasets import get_dataset, infer_candidate_facets


def test_paper_dataset_registry_contains_mixed_modality_dataset():
    spec = get_dataset("mmdocrag")

    assert spec.hf_repo == "MMDocIR/MMDocRAG"
    assert spec.primary_facets == ("mixed_modality_order",)


def test_infer_candidate_facets_from_features():
    facets = infer_candidate_facets(
        feature_names={"question", "options", "images", "evidence_list"},
        modalities={"image", "text"},
        task_categories={"multimodal"},
    )

    assert "option_order" in facets
    assert "image_set_order" in facets
    assert "mixed_modality_order" in facets


def test_infer_candidate_facets_from_nested_feature_paths():
    facets = infer_candidate_facets(
        feature_names={
            "question.choices.text",
            "retrieved_documents[].rank",
            "retrieved_documents[].passage",
        },
        modalities={"text"},
        task_categories={"question-answering"},
    )

    assert "option_order" in facets
    assert "evidence_chunk_order" in facets
    assert "document_rank_order" in facets
