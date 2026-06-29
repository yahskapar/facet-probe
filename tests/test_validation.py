from facet_probe.validation import validate_audit_items


def test_validate_audit_items_accepts_valid_option_item():
    report = validate_audit_items(
        [
            {
                "dataset": "toy",
                "item_id": "i1",
                "choices": ["A", "B"],
                "gold": "0",
                "components": [
                    {"component_id": "c0", "kind": "choice", "content_ref": "choice:0"},
                    {"component_id": "c1", "kind": "choice", "content_ref": "choice:1"},
                ],
            }
        ],
        facet="option_order",
    )

    assert report.ok
    assert report.n_items == 1
    assert report.n_components == 2


def test_validate_audit_items_flags_duplicates_and_empty_components():
    report = validate_audit_items(
        [
            {"dataset": "toy", "item_id": "i1", "components": []},
            {"dataset": "toy", "item_id": "i1", "components": []},
        ],
        facet="evidence_chunk_order",
    )

    assert not report.ok
    codes = {issue.code for issue in report.issues}
    assert "duplicate_item_id" in codes
    assert "empty_components" in codes
