import pytest

import facet_probe.runner as runner
from facet_probe.profiles import DatasetProfile, EvaluationProfile, ModelProfile
from facet_probe.runner import (
    MockAdapter,
    PromptImagePiece,
    PromptTextPiece,
    RuntimeExample,
    _build_prompt_bundle,
    _component_to_piece,
    _execute_trial,
    _hf_attention_kwargs_variants,
    _load_hf_with_fast_fallback,
    _mcq_example,
    execute_profile,
)
from facet_probe.schema import AuditItem, Component


class FakeImage:
    mode = "RGB"

    def convert(self, _mode):
        return self

    def save(self, _buf, format=None):
        return None


class FakeCuda:
    def __init__(self, available: bool):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorch:
    def __init__(self, cuda_available: bool):
        self.cuda = FakeCuda(cuda_available)


def test_hf_fast_mode_prefers_flash_then_sdpa_then_default(monkeypatch):
    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: object() if name == "flash_attn" else None,
    )

    variants = _hf_attention_kwargs_variants(
        {"fast_mode": "auto"},
        FakeTorch(cuda_available=True),
    )

    assert variants == (
        {"attn_implementation": "flash_attention_2"},
        {"attn_implementation": "sdpa"},
        {},
    )


def test_hf_fast_mode_skips_flash_without_cuda(monkeypatch):
    monkeypatch.setattr(
        runner.importlib.util,
        "find_spec",
        lambda name: object() if name == "flash_attn" else None,
    )

    variants = _hf_attention_kwargs_variants(
        {"fast_mode": "auto"},
        FakeTorch(cuda_available=False),
    )

    assert variants == ({"attn_implementation": "sdpa"}, {})


def test_hf_fast_mode_can_be_disabled_or_required(monkeypatch):
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _name: None)

    assert _hf_attention_kwargs_variants({"fast_mode": "off"}, FakeTorch(True)) == ({},)
    assert _hf_attention_kwargs_variants({"fast_mode": "require"}, FakeTorch(False)) == (
        {"attn_implementation": "flash_attention_2"},
    )
    assert _hf_attention_kwargs_variants(
        {"attn_implementation": "eager"},
        FakeTorch(True),
    ) == ({"attn_implementation": "eager"},)


def test_hf_fast_loader_warns_and_falls_back():
    attempts = []

    def loader(kwargs):
        attempts.append(dict(kwargs))
        if kwargs.get("attn_implementation") == "flash_attention_2":
            raise RuntimeError("flash unavailable")
        return {"loaded_with": dict(kwargs)}

    with pytest.warns(RuntimeWarning, match="fast load attempt"):
        loaded = _load_hf_with_fast_fallback(
            loader,
            ({"attn_implementation": "flash_attention_2"}, {}),
            description="test model",
        )

    assert loaded == {"loaded_with": {}}
    assert attempts == [{"attn_implementation": "flash_attention_2"}, {}]


def test_prompt_bundle_preserves_image_pieces():
    item = AuditItem(
        item_id="toy::image",
        dataset="toy",
        components=(
            Component("image_0", "image", "toy-image-0", "Image 1"),
            Component("image_1", "image", "toy-image-1", "Image 2"),
        ),
        question_ref="toy-question",
        choices=("left", "right"),
        gold="A",
    )
    example = RuntimeExample(
        facet="image_set_order",
        item=item,
        question="Which image is the target?",
        content={"toy-image-0": FakeImage(), "toy-image-1": FakeImage()},
        score_kind="mcq_letter",
        gold_normalized="A",
    )

    bundle = _build_prompt_bundle(
        example,
        {
            "ordered_component_ids": ["image_1", "image_0"],
            "permutation": [1, 0],
        },
    )

    assert isinstance(bundle.pieces[0], PromptImagePiece)
    assert isinstance(bundle.pieces[1], PromptImagePiece)
    assert "Question: Which image is the target?" in bundle.text


def test_mcq_rows_with_images_keep_image_as_fixed_context():
    dataset = DatasetProfile(
        name="toy_vqa",
        hf_repo="toy/vqa",
        split="validation",
        facets=("option_order",),
        filters={"min_choices": 2, "requires_image": True},
    )

    example = _mcq_example(
        {
            "id": "v1",
            "question": "What color is shown?",
            "choices": ["red", "blue"],
            "answer": "B",
            "image": FakeImage(),
        },
        dataset,
        "option_order",
        0,
    )
    bundle = _build_prompt_bundle(
        example,
        {
            "ordered_component_ids": ["choice_1", "choice_0"],
            "permutation": [1, 0],
        },
    )

    assert [component.component_id for component in example.item.components] == [
        "choice_0",
        "choice_1",
    ]
    assert [component.component_id for component in example.fixed_components] == [
        "fixed_image_0"
    ]
    assert isinstance(bundle.pieces[0], PromptImagePiece)
    assert "  A) blue\n  B) red" in bundle.text


def test_text_components_are_not_coerced_as_image_paths():
    long_text = "Meet Corliss Archer: " + ("radio program " * 40)
    component = Component("text_0", "text", "long-text", "Evidence")

    piece = _component_to_piece(component, {"long-text": long_text})

    assert isinstance(piece, PromptTextPiece)
    assert piece.text == long_text


def test_execute_profile_fails_on_audited_count_shortfall(monkeypatch, tmp_path):
    def one_row(_dataset, *, streaming=True):
        del streaming
        yield {
            "id": "q1",
            "question": "Pick blue.",
            "choices": ["red", "blue"],
            "answer": "B",
        }

    monkeypatch.setattr(runner, "_iter_hf_dataset_rows", one_row)
    profile = EvaluationProfile(
        name="shortfall-test",
        models=(
            ModelProfile(
                name="deterministic-mock",
                provider="mock",
                model_id="deterministic-mock",
                adapter="provider_api",
            ),
        ),
        datasets=(
            DatasetProfile(
                name="toy_mcq",
                hf_repo="toy/mcq",
                split="validation",
                facets=("option_order",),
                audited_n=2,
            ),
        ),
        k_orderings=1,
    )

    with pytest.raises(RuntimeError, match="audited item counts"):
        execute_profile(profile, tmp_path / "strict")

    status = execute_profile(profile, tmp_path / "partial", allow_partial=True)

    assert status["status"] == "completed"
    assert status["shortfalls"] == [
        {"dataset": "toy_mcq", "facet": "option_order", "expected": 2, "loaded": 1}
    ]


def test_mixed_modality_trial_records_include_question_for_judge():
    item = AuditItem(
        item_id="mix::1",
        dataset="mix",
        components=(
            Component("text_0", "text", "mix-text-0", "Text"),
            Component("image_1", "image", "mix-image-1", "Image"),
        ),
        question_ref="mix-question",
        gold="flour",
    )
    example = RuntimeExample(
        facet="mixed_modality_order",
        item=item,
        question="What ingredient should be added?",
        content={"mix-text-0": "Beat the eggs.", "mix-image-1": FakeImage()},
        score_kind="exact_match",
        gold_normalized="flour",
    )

    record = _execute_trial(
        adapter=MockAdapter(
            ModelProfile(
                name="deterministic-mock",
                provider="mock",
                model_id="deterministic-mock",
                adapter="provider_api",
            )
        ),
        model=ModelProfile(
            name="deterministic-mock",
            provider="mock",
            model_id="deterministic-mock",
            adapter="provider_api",
        ),
        example=example,
        manifest_row={
            "ordered_component_ids": ["text_0", "image_1"],
            "permutation": [0, 1],
            "ordering_idx": 0,
        },
        max_new_tokens=8,
        include_raw_outputs=True,
    )

    assert record["question"] == "What ingredient should be added?"
