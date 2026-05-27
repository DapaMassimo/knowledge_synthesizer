import pytest

from knowledge_synthesizer.adapters.retrieval.query_transformers import (
    HydeTransformer,
    MultiQueryTransformer,
    PassthroughTransformer,
)
from knowledge_synthesizer.application.indexing import IndexingService
from knowledge_synthesizer.application.qa import QAService
from knowledge_synthesizer.application.summarization import SummarizationService
from knowledge_synthesizer.composition.container import Container
from knowledge_synthesizer.config.settings import Settings

pytestmark = pytest.mark.unit


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "chroma_path": None,
        "docling_cache_dir": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_container_builds_use_cases() -> None:
    container = Container(_settings())
    assert isinstance(container.qa_service(), QAService)
    assert isinstance(container.summarization_service(), SummarizationService)
    assert isinstance(container.indexing_service(), IndexingService)


def test_vector_store_and_cache_are_memoized() -> None:
    container = Container(_settings())
    assert container._vector_store() is container._vector_store()
    assert container._docling_cache() is container._docling_cache()


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("passthrough", PassthroughTransformer),
        ("hyde", HydeTransformer),
        ("multiquery", MultiQueryTransformer),
    ],
)
def test_query_strategy_selects_transformer(strategy: str, expected: type) -> None:
    container = Container(_settings(query_strategy=strategy))
    assert isinstance(container._transformer(), expected)
