from facet_probe.facets import sample_permutations, source_option_index


def test_sample_permutations_is_stable_and_canonical_first():
    a = sample_permutations(5, item_id="item-1", seed=42)
    b = sample_permutations(5, item_id="item-1", seed=42)

    assert a == b
    assert a[0] == (0, 1, 2, 3, 4)
    assert len(a) == 6
    assert len(set(a)) == 6


def test_sample_permutations_cycles_when_factorial_lt_k():
    perms = sample_permutations(2, item_id="small", k=6)

    assert len(perms) == 6
    assert set(perms) == {(0, 1), (1, 0)}


def test_source_option_index_maps_display_letter_to_source_content():
    # Display slot A contains source option 2, B contains source option 0, etc.
    permutation = (2, 0, 1)

    assert source_option_index("A", permutation) == "2"
    assert source_option_index("B", permutation) == "0"
    assert source_option_index("C", permutation) == "1"
    assert source_option_index("D", permutation) is None
