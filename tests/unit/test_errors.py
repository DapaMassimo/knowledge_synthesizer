import pytest

from knowledge_synthesizer.domain.errors import (
    ChunkingError,
    ConfigurationError,
    EmbeddingError,
    KnowledgeSynthesizerError,
    LLMError,
    ParsingError,
    RetrievalError,
    SourceLoadError,
    VectorStoreError,
)

pytestmark = pytest.mark.unit

_SUBCLASSES = [
    SourceLoadError,
    ParsingError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
    RetrievalError,
    LLMError,
    ConfigurationError,
]


@pytest.mark.parametrize("exc_type", _SUBCLASSES)
def test_domain_errors_share_a_common_base(exc_type: type[KnowledgeSynthesizerError]) -> None:
    assert issubclass(exc_type, KnowledgeSynthesizerError)
    with pytest.raises(KnowledgeSynthesizerError):
        raise exc_type("boom")
