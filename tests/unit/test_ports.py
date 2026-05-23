import pytest

from knowledge_synthesizer.domain.ports import (
    EmbeddingModel,
    LLMProvider,
    VectorStore,
)
from tests.fakes import FakeEmbeddings, FakeLLM, InMemoryVectorStore

pytestmark = pytest.mark.unit


def test_fakes_satisfy_their_ports_structurally() -> None:
    assert isinstance(FakeLLM(), LLMProvider)
    assert isinstance(FakeEmbeddings(), EmbeddingModel)
    assert isinstance(InMemoryVectorStore(), VectorStore)


def test_unrelated_object_does_not_satisfy_a_port() -> None:
    assert not isinstance(object(), LLMProvider)
