from facet_probe.templates import (
    evidence_list_audit_item,
    image_list_audit_item,
    mcq_audit_item,
    mixed_modality_audit_item,
    render_ordered_text_prompt,
)


def test_mcq_template_handles_hf_choice_mapping():
    item = mcq_audit_item(
        {
            "id": "q1",
            "question": "Which color?",
            "choices": {"label": ["A", "B"], "text": ["red", "blue"]},
            "answer": "B",
        },
        dataset="toy_mcq",
    )

    assert item.item_id == "toy_mcq::q1"
    assert item.choices == ("red", "blue")
    assert item.gold == "B"
    assert [component.component_id for component in item.components] == ["choice_0", "choice_1"]
    assert item.components[0].content_ref == "facet-probe://toy_mcq/q1/choices/0"


def test_evidence_and_image_templates_create_content_refs_without_raw_content():
    evidence_item = evidence_list_audit_item(
        {"id": "e1", "evidence": [{"text": "alpha"}, {"text": "beta"}], "answer": "alpha"},
        dataset="toy_evidence",
    )
    image_item = image_list_audit_item(
        {"id": "i1", "images": ["image-a.png", "image-b.png"], "answer": "A"},
        dataset="toy_images",
    )

    assert [component.kind for component in evidence_item.components] == ["text", "text"]
    assert [component.content_ref for component in evidence_item.components] == [
        "facet-probe://toy_evidence/e1/evidence/0",
        "facet-probe://toy_evidence/e1/evidence/1",
    ]
    assert [component.kind for component in image_item.components] == ["image", "image"]
    assert image_item.components[1].content_ref == "facet-probe://toy_images/i1/images/1"


def test_mixed_modality_template_and_prompt_renderer():
    item = mixed_modality_audit_item(
        {
            "id": "m1",
            "captions": ["first caption", "second caption"],
            "images": ["image-a", "image-b"],
            "answer": "done",
        },
        dataset="toy_mixed",
        component_fields=(("captions", "text"), ("images", "image")),
    )
    content_by_ref = {
        component.content_ref: f"resolved:{component.component_id}"
        for component in item.components
    }

    prompt = render_ordered_text_prompt(
        item,
        ["images_1", "captions_0"],
        question="Summarize.",
        resolve_content=lambda component: content_by_ref[component.content_ref],
    )

    assert "Question: Summarize." in prompt
    assert "[1] images 2: resolved:images_1" in prompt
    assert "[2] captions 1: resolved:captions_0" in prompt


def test_prompt_renderer_reorders_mcq_choices():
    item = mcq_audit_item(
        {
            "id": "q2",
            "question": "Pick one.",
            "choices": ["red", "blue", "green"],
            "answer": "0",
        },
        dataset="toy_mcq",
    )

    prompt = render_ordered_text_prompt(
        item,
        ["choice_2", "choice_0", "choice_1"],
        question="Pick one.",
        resolve_content=lambda component: component.content_ref,
    )

    assert "  A) green\n  B) red\n  C) blue" in prompt
    assert "Evidence:" not in prompt
