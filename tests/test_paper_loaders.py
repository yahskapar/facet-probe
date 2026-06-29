import base64
import gzip
import json
import zipfile
from pathlib import Path

import facet_probe.paper_loaders as paper_loaders
from facet_probe.profiles import DatasetProfile, EvaluationProfile, ModelProfile, paper_profile
from facet_probe.runner import RuntimeExample, _build_prompt_bundle, execute_profile
from facet_probe.schema import AuditItem, Component

REPO_ROOT = Path(__file__).resolve().parents[1]
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


class FakeImage:
    mode = "RGB"

    def convert(self, _mode):
        return self

    def save(self, _buf, format=None):
        return None


def test_paper_loader_registry_covers_release_profile():
    profile = paper_profile(config_dir=REPO_ROOT / "configs", models="qwen3-5-4b")

    missing = [
        (facet, dataset.name)
        for dataset in profile.datasets
        for facet in dataset.facets
        if (facet, dataset.name) not in paper_loaders._LOADERS
    ]

    assert missing == []


def test_paper_loader_respects_streaming_flag(monkeypatch):
    seen = {}

    def fake_load_dataset(*_args, **kwargs):
        seen["streaming"] = kwargs.get("streaming")
        return [
            {
                "question_id": "q1",
                "question": "Pick the blue option.",
                "options": ["red", "blue", "green", "yellow"],
                "answer_index": 1,
            }
        ]

    monkeypatch.setattr(paper_loaders, "_load_dataset", fake_load_dataset)
    dataset = DatasetProfile(
        name="mmlu_pro",
        hf_repo="TIGER-Lab/MMLU-Pro",
        split="test",
        facets=("option_order",),
        audited_n=1,
    )

    examples = paper_loaders.load_paper_dataset_examples(dataset, target=1, streaming=False)

    assert seen["streaming"] is False
    assert len(examples) == 1


def test_mmlu_pro_loader_normalizes_mcq_rows(monkeypatch):
    monkeypatch.setattr(
        paper_loaders,
        "_load_dataset",
        lambda *args, **kwargs: [
            {
                "question_id": "q1",
                "question": "Pick the blue option.",
                "options": ["red", "blue", "green", "yellow"],
                "answer_index": 1,
                "category": "toy",
            }
        ],
    )
    dataset = DatasetProfile(
        name="mmlu_pro",
        hf_repo="TIGER-Lab/MMLU-Pro",
        split="test",
        facets=("option_order",),
        audited_n=1,
    )

    examples = paper_loaders.load_paper_dataset_examples(dataset, target=1)

    assert examples is not None
    assert len(examples) == 1
    assert examples[0].item.choices == ("red", "blue", "green", "yellow")
    assert examples[0].gold_normalized == "1"


def test_option_order_loaders_for_commonsenseqa_and_mathvision(monkeypatch):
    def fake_load_dataset(repo, *args, **kwargs):
        del args, kwargs
        if repo == "tau/commonsense_qa":
            return [
                {
                    "id": "cs1",
                    "question": "What do people use to write?",
                    "choices": {"label": ["A", "B"], "text": ["pen", "spoon"]},
                    "answerKey": "A",
                }
            ]
        if repo == "MathLLMs/MathVision":
            return [
                {
                    "id": "mv1",
                    "question": "What is shown?",
                    "options": ["1", "2", "3"],
                    "answer": "C",
                    "image": FakeImage(),
                }
            ]
        raise AssertionError(repo)

    monkeypatch.setattr(paper_loaders, "_load_dataset", fake_load_dataset)

    csqa = paper_loaders.load_option_order_commonsenseqa(
        _dataset("commonsenseqa", "option_order"),
        target=1,
    )
    mathvision = paper_loaders.load_option_order_mathvision(
        _dataset("mathvision", "option_order"),
        target=1,
    )

    assert csqa[0].item.choices == ("pen", "spoon")
    assert csqa[0].gold_normalized == "0"
    assert mathvision[0].gold_normalized == "2"
    assert mathvision[0].fixed_components[0].kind == "image"


def test_text_evidence_and_document_loaders(monkeypatch):
    hotpot_row = {
        "id": "hp1",
        "question": "Who founded it?",
        "answer": "Ada",
        "context": {
            "title": ["A", "B", "C"],
            "sentences": [["Ada founded it."], ["Other fact."], ["More context."]],
        },
        "supporting_facts": {"title": ["A"]},
    }
    musique_row = {
        "id": "mu1",
        "question": "What is the answer?",
        "answer": "Paris",
        "paragraphs": [
            {"title": "p1", "paragraph_text": "Paris clue.", "is_supporting": True},
            {"title": "p2", "paragraph_text": "France clue.", "is_supporting": True},
            {"title": "p3", "paragraph_text": "Distractor.", "is_supporting": False},
        ],
    }
    multihop_row = {
        "query": "Where was she born?",
        "answer": "London",
        "evidence_list": [
            {"title": "d1", "fact": "She was born in London."},
            {"title": "d2", "fact": "She later moved."},
        ],
    }

    def fake_load_dataset(repo=None, *args, **kwargs):
        del args, kwargs
        if repo == "dgslibisey/MuSiQue":
            return [musique_row]
        if repo == "yixuantt/MultiHopRAG":
            return [multihop_row]
        raise AssertionError(repo)

    monkeypatch.setattr(
        paper_loaders,
        "_load_hotpotqa_val_ds",
        lambda **_kwargs: [hotpot_row],
    )
    monkeypatch.setattr(paper_loaders, "_load_dataset", fake_load_dataset)

    hotpot_evidence = paper_loaders.load_evidence_chunk_order_hotpotqa(
        _dataset("hotpotqa", "evidence_chunk_order"),
        target=1,
    )
    musique = paper_loaders.load_evidence_chunk_order_musique(
        _dataset("musique", "evidence_chunk_order"),
        target=1,
    )
    multihop = paper_loaders.load_document_rank_order_multihop_rag(
        _dataset("multihop_rag", "document_rank_order"),
        target=1,
    )
    hotpot_docs = paper_loaders.load_document_rank_order_hotpotqa(
        _dataset("hotpotqa", "document_rank_order"),
        target=1,
    )

    assert hotpot_evidence[0].gold_normalized == "ada"
    assert musique[0].gold_normalized == "paris"
    assert multihop[0].item.components[0].kind == "document"
    assert hotpot_docs[0].item.components[0].label == "Retrieved doc rank 1"


def test_medxpertqa_loader_rebuilds_manifest_shape(monkeypatch):
    monkeypatch.setattr(
        paper_loaders,
        "_load_medxpertqa_records",
        lambda *_args: [
            {
                "id": "mx1",
                "question": (
                    "History: cough\n"
                    "Laboratory: high count\n"
                    "Imaging: opacity\n"
                    "What is the diagnosis?"
                ),
                "images": [FakeImage()],
                "options": ["pneumonia", "flu"],
                "answer": "A",
                "specialty": "pulmonology",
            }
        ],
    )

    examples = paper_loaders.load_evidence_chunk_order_medxpertqa(
        _dataset("medxpertqa", "evidence_chunk_order", config="MM"),
        target=1,
    )

    assert examples[0].score_kind == "mcq_letter"
    assert examples[0].gold_normalized == "A"
    assert len(examples[0].fixed_components) == 1


def test_image_set_loaders_for_mantis_and_medframe(monkeypatch):
    def fake_load_dataset(repo, *args, **kwargs):
        del args, kwargs
        if repo == "TIGER-Lab/Mantis-Eval":
            return [
                {
                    "id": "ma1",
                    "question_type": "multi-choice",
                    "question": "<image> Which image is target?",
                    "images": [FakeImage(), FakeImage(), FakeImage()],
                    "options": ["(A) left", "(B) right"],
                    "answer": "B",
                }
            ]
        if repo == "SuhaoYu1020/MedFrameQA":
            return [
                {
                    "question_id": "mf1",
                    "question": "Which frame?",
                    "image_0": FakeImage(),
                    "image_1": FakeImage(),
                    "image_2": FakeImage(),
                    "options": {"A": "first", "B": "second"},
                    "correct_answer": "A",
                }
            ]
        raise AssertionError(repo)

    monkeypatch.setattr(paper_loaders, "_load_dataset", fake_load_dataset)

    mantis = paper_loaders.load_image_set_order_mantis_eval(
        _dataset("mantis_eval", "image_set_order"),
        target=1,
    )
    medframe = paper_loaders.load_image_set_order_medframeqa(
        _dataset("medframeqa", "image_set_order"),
        target=1,
    )

    assert mantis[0].item.choices == ("left", "right")
    assert mantis[0].gold_normalized == "B"
    assert len(medframe[0].item.components) == 3


def test_mramg_loader_from_public_files(monkeypatch, tmp_path):
    qa = tmp_path / "recipe_mqa.jsonl"
    doc = tmp_path / "doc_recipe.jsonl"
    img_zip = tmp_path / "IMAGE.zip"
    qa.write_text(
        json.dumps(
            {
                "id": "r1",
                "question": "What is next?",
                "ground_truth": "flour",
                "provenance": [7],
                "images_list": ["a", "b", "c"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    doc.write_text(
        json.dumps([7, "Text one <PIC> Text two <PIC> Text three <PIC> Text four"]) + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(img_zip, "w") as zf:
        for name in ("a", "b", "c"):
            zf.writestr(f"IMAGE/images/RECIPE/{name}.jpg", PNG_BYTES)

    monkeypatch.setattr(
        paper_loaders,
        "_hf_download",
        lambda _repo, filename: {
            "recipe_mqa.jsonl": qa,
            "doc_recipe.jsonl": doc,
            "IMAGE.zip": img_zip,
        }[filename],
    )

    examples = paper_loaders.load_mixed_modality_order_mramg(
        _dataset("mramg", "mixed_modality_order"),
        target=1,
    )

    assert len(examples) == 1
    assert examples[0].facet == "mixed_modality_order"
    assert examples[0].gold_normalized == "flour"


def test_mmdocrag_loader_from_public_files(monkeypatch, tmp_path):
    qa = tmp_path / "dev_15.jsonl"
    img_zip = tmp_path / "images.zip"
    qa.write_text(
        json.dumps(
            {
                "id": "md1",
                "doc_id": "doc1",
                "question": "What value?",
                "answer_short": "0.16",
                "text_quotes": [{"text": "a"}, {"text": "b"}, {"text": "c"}],
                "img_quotes": [
                    {"img_path": "i0.png"},
                    {"img_path": "i1.png"},
                    {"img_path": "i2.png"},
                ],
                "gold_quotes": ["text0", "image0"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(img_zip, "w") as zf:
        for name in ("i0.png", "i1.png", "i2.png"):
            zf.writestr(name, PNG_BYTES)

    monkeypatch.setattr(
        paper_loaders,
        "_hf_download",
        lambda _repo, filename: {"dev_15.jsonl": qa, "images.zip": img_zip}[filename],
    )

    examples = paper_loaders.load_mixed_modality_order_mmdocrag(
        _dataset("mmdocrag", "mixed_modality_order"),
        target=1,
    )

    assert len(examples) == 1
    assert examples[0].gold_normalized == "0.16"


def test_mmqa_loader_from_public_files(monkeypatch, tmp_path):
    qa = tmp_path / "MMQA_dev.jsonl.gz"
    texts = tmp_path / "MMQA_texts.jsonl.gz"
    images = tmp_path / "MMQA_images.jsonl.gz"
    image_zip = tmp_path / "mmqa_images.zip"
    _write_gzip_jsonl(
        qa,
        [
            {
                "qid": "q1",
                "question": "What year?",
                "metadata": {
                    "text_doc_ids": ["t1", "t2", "t3"],
                    "image_doc_ids": ["i1", "i2", "i3"],
                },
                "answers": [{"answer": "1995"}],
            }
        ],
    )
    _write_gzip_jsonl(
        texts,
        [
            {"id": "t1", "text": "one"},
            {"id": "t2", "text": "two"},
            {"id": "t3", "text": "three"},
        ],
    )
    _write_gzip_jsonl(
        images,
        [
            {"id": "i1", "path": "i1.png"},
            {"id": "i2", "path": "i2.png"},
            {"id": "i3", "path": "i3.png"},
        ],
    )
    with zipfile.ZipFile(image_zip, "w") as zf:
        for name in ("i1.png", "i2.png", "i3.png"):
            zf.writestr(name, PNG_BYTES)

    monkeypatch.setattr(
        paper_loaders,
        "_hf_download",
        lambda _repo, filename: {
            "MMQA_dev.jsonl.gz": qa,
            "MMQA_texts.jsonl.gz": texts,
            "MMQA_images.jsonl.gz": images,
        }[filename],
    )
    monkeypatch.setattr(paper_loaders, "_ensure_mmqa_image_zip", lambda _cache_dir: image_zip)

    examples = paper_loaders.load_mixed_modality_order_mmqa(
        _dataset("mmqa", "mixed_modality_order"),
        target=1,
    )

    assert len(examples) == 1
    assert examples[0].gold_normalized == "1995"


def test_runner_uses_paper_loader_before_generic_hf(monkeypatch, tmp_path):
    def loader(dataset, target):
        assert dataset.name == "paper_toy"
        assert target == 1
        item = AuditItem(
            item_id="paper_toy::1",
            dataset="paper_toy",
            components=(
                Component("choice_0", "choice", "toy-choice-a", "A"),
                Component("choice_1", "choice", "toy-choice-b", "B"),
            ),
            question_ref="toy-question",
            choices=("red", "blue"),
            gold="1",
        )
        return [
            RuntimeExample(
                facet="option_order",
                item=item,
                question="Pick blue.",
                content={"toy-choice-a": "red", "toy-choice-b": "blue"},
                score_kind="option_content_idx",
                gold_normalized="1",
            )
        ]

    def fail_generic(*_args, **_kwargs):
        raise AssertionError("generic HF loader should not be called")

    monkeypatch.setitem(paper_loaders._LOADERS, ("option_order", "paper_toy"), loader)
    monkeypatch.setattr("facet_probe.runner._iter_hf_dataset_rows", fail_generic)
    profile = EvaluationProfile(
        name="paper-loader-test",
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
                name="paper_toy",
                hf_repo="toy/paper",
                split="validation",
                facets=("option_order",),
                audited_n=1,
            ),
        ),
        k_orderings=2,
    )

    status = execute_profile(profile, tmp_path / "run")

    assert status["status"] == "completed"
    assert status["n_manifest_rows"] == 2
    assert status["shortfalls"] == []
    assert (tmp_path / "run" / "trials.jsonl").exists()


def test_mixed_modality_examples_use_free_form_prompting():
    example = paper_loaders._make_mixed_modality_example(
        dataset=DatasetProfile(
            name="mramg",
            hf_repo="MRAMG/MRAMG-Bench",
            split="recipe_mqa.jsonl",
            facets=("mixed_modality_order",),
        ),
        raw_id="mix1",
        question="What should be added next?",
        components_raw=(
            {"kind": "text", "text": "Beat the eggs."},
            {"kind": "image", "slot_idx": 0},
        ),
        images={0: FakeImage()},
        gold="flour",
    )

    assert example is not None
    assert example.score_kind == "exact_match"
    assert "free-form" in example.answer_instruction
    assert "single letter" in example.system_instruction

    bundle = _build_prompt_bundle(
        example,
        {
            "ordered_component_ids": ["text_0", "image_1"],
            "permutation": [0, 1],
        },
    )

    assert "Write a concise free-form answer" in bundle.text


def _dataset(name: str, facet: str, *, config: str | None = None) -> DatasetProfile:
    return DatasetProfile(
        name=name,
        hf_repo=f"fake/{name}",
        split="test",
        facets=(facet,),
        config=config,
        audited_n=1,
    )


def _write_gzip_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
