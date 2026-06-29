from facet_probe.scoring import normalize_answer, normalize_text, parse_answer_letter, score_answer


def test_parse_answer_letter_prefers_answer_line():
    raw = "Reasoning goes here.\nAnswer: C"

    assert parse_answer_letter(raw, 4) == "C"


def test_option_content_normalization_uses_permutation():
    raw = "Answer: B"
    permutation = (3, 1, 0, 2)

    assert normalize_answer("option_content_idx", raw, n_choices=4, permutation=permutation) == "1"


def test_normalize_text_strips_articles_and_answer_prefix():
    assert normalize_text("Answer: The Eiffel Tower.") == "eiffel tower"


def test_score_answer_handles_missing_values():
    assert score_answer("a", "a") is True
    assert score_answer("a", "b") is False
    assert score_answer(None, "b") is None
