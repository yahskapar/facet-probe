from facet_probe.manifests import trial_manifest_rows


def test_trial_manifest_rows_emit_ordered_component_ids():
    items = [
        {
            "item_id": "toy-1",
            "dataset": "toy",
            "components": [
                {"component_id": "a", "kind": "choice", "content_ref": "choice:a"},
                {"component_id": "b", "kind": "choice", "content_ref": "choice:b"},
                {"component_id": "c", "kind": "choice", "content_ref": "choice:c"},
            ],
            "gold": "a",
        }
    ]

    rows = trial_manifest_rows(items, facet="option_order", k=2, seed=42)

    assert len(rows) == 2
    assert rows[0]["permutation"] == [0, 1, 2]
    assert rows[0]["ordered_component_ids"] == ["a", "b", "c"]
    assert rows[0]["gold"] == "a"


def test_trial_manifest_rows_can_include_ordered_components():
    items = [
        {
            "item_id": "toy-1",
            "dataset": "toy",
            "components": [{"component_id": "a", "kind": "text", "content_ref": "text:a"}],
        }
    ]

    rows = trial_manifest_rows(
        items,
        facet="evidence_chunk_order",
        k=1,
        include_ordered_components=True,
    )

    assert rows[0]["ordered_components"][0]["component_id"] == "a"
