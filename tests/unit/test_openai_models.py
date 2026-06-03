import pytest

from knowledge_synthesizer.adapters.llm.openai_models import classify_models

pytestmark = pytest.mark.unit


def test_classify_models_splits_llms_and_embeddings() -> None:
    ids = [
        "gpt-4o",
        "gpt-4o-mini",
        "o3-mini",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "gpt-4o-audio-preview",  # excluded (audio)
        "dall-e-3",  # excluded
        "whisper-1",  # excluded
        "omni-moderation-latest",  # excluded (not an llm prefix)
        "babbage-002",  # excluded (not an llm prefix)
    ]
    llms, embeddings = classify_models(ids)
    assert llms == ["gpt-4o", "gpt-4o-mini", "o3-mini"]
    assert embeddings == ["text-embedding-3-large", "text-embedding-3-small"]


def test_classify_models_dedupes() -> None:
    llms, _ = classify_models(["gpt-4o", "gpt-4o"])
    assert llms == ["gpt-4o"]
