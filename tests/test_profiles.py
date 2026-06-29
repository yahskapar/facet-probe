from pathlib import Path

import facet_probe as fp

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_paper_profile_loads_selected_configs():
    profile = fp.paper_profile(
        config_dir=REPO_ROOT / "configs",
        models=["gemini-3.1-pro-preview", "qwen3-5-4b"],
        datasets=["mmlu_pro", "mmqa"],
    )

    assert profile.k_orderings == 6
    assert profile.seed == 42
    assert [model.name for model in profile.models] == [
        "gemini-3.1-pro-preview",
        "qwen3-5-4b",
    ]
    assert profile.models[0].provider == "google"
    assert profile.models[0].adapter == "provider_api"
    assert profile.models[1].provider == "huggingface"
    assert profile.models[1].adapter == "huggingface_local"
    assert [dataset.name for dataset in profile.datasets] == ["mmlu_pro", "mmqa"]
    assert profile.datasets[0].facets == ("option_order",)


def test_default_paper_profile_loads_from_outside_repo_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    profile = fp.paper_profile(models="qwen3-5-4b", datasets="mmlu_pro")

    assert profile.models[0].name == "qwen3-5-4b"
    assert profile.datasets[0].name == "mmlu_pro"
    assert profile.k_orderings == 6


def test_paper_profile_has_public_split_labels():
    profile = fp.paper_profile(config_dir=REPO_ROOT / "configs", models="qwen3-5-4b")

    splits = {dataset.name: dataset.split for dataset in profile.datasets}
    assert "REVIEW_SPLIT" not in splits.values()
    assert splits["mramg"] == "file-backed"
    assert splits["mmdocrag"] == "file-backed"


def test_judge_profile_loads_primary_mixed_semantic_judge(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    judge = fp.judge_profile(config_dir=REPO_ROOT / "configs")

    assert judge.name == "mixed-semantic-primary"
    assert judge.provider == "google"
    assert judge.model_id == "gemini-3.1-pro-preview"
    assert judge.env == ("GOOGLE_API_KEY",)
    assert judge.generation["max_output_tokens"] == 512
    assert judge.env_status()["ok"] is False

    monkeypatch.setenv("GOOGLE_API_KEY", "not-a-real-key")
    assert judge.env_status()["ok"] is True


def test_closed_and_hf_models_share_common_profile_shape():
    closed = fp.model_profile(
        "openai",
        "gpt-example",
        generation={"temperature": 0, "top_p": 1},
    )
    local = fp.model_profile(
        "huggingface",
        "Qwen/Qwen3.5-VL-4B-Instruct",
        dtype="bfloat16",
        load_in_4bit=True,
    )

    assert closed.to_dict()["provider"] == "openai"
    assert local.to_dict()["provider"] == "huggingface"
    assert closed.to_dict().keys() == local.to_dict().keys()
    assert closed.env == ("OPENAI_API_KEY",)
    assert local.generation["load_in_4bit"] is True


def test_custom_model_env_status_uses_profile_env(monkeypatch):
    monkeypatch.delenv("MY_PROVIDER_KEY", raising=False)
    model = fp.model_profile(
        "openai",
        "gpt-example",
        api_key_env="MY_PROVIDER_KEY",
    )

    assert model.env_status()["ok"] is False
    monkeypatch.setenv("MY_PROVIDER_KEY", "secret")
    assert model.env_status()["ok"] is True


def test_profile_can_add_or_replace_new_hf_dataset():
    paper = fp.paper_profile(
        config_dir=REPO_ROOT / "configs",
        models="gemini-3.1-pro-preview",
        datasets="mmlu_pro",
    )
    new_dataset = fp.hf_dataset(
        "https://huggingface.co/datasets/allenai/ai2_arc",
        name="arc_challenge",
        config="ARC-Challenge",
        split="validation",
        facets=("option_order",),
        filters={"min_choices": 4},
    )

    combined = paper.add_datasets(new_dataset, name="paper-plus-arc-challenge")
    custom_only = paper.only_datasets(new_dataset, name="arc-challenge-only")

    assert [dataset.name for dataset in combined.datasets] == ["mmlu_pro", "arc_challenge"]
    assert [dataset.name for dataset in custom_only.datasets] == ["arc_challenge"]
    assert combined.to_dict()["datasets"][1]["hf_repo"] == "allenai/ai2_arc"
