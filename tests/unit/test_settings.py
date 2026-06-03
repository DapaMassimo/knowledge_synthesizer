import pytest

from knowledge_synthesizer.config.settings import Settings

pytestmark = pytest.mark.unit


def test_model_lists_parse_comma_separated_strings() -> None:
    settings = Settings(llm_models="gpt-4o, o3-mini ,gpt-4o-mini", embedding_models="a,b")
    assert settings.llm_models == ["gpt-4o", "o3-mini", "gpt-4o-mini"]
    assert settings.embedding_models == ["a", "b"]


def test_model_lists_accept_python_lists() -> None:
    settings = Settings(llm_models=["x", "y"])
    assert settings.llm_models == ["x", "y"]


def test_model_lists_have_sensible_defaults() -> None:
    settings = Settings()
    assert "gpt-4o-mini" in settings.llm_models
    assert "text-embedding-3-small" in settings.embedding_models
