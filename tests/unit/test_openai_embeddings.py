from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError

from knowledge_synthesizer.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
from knowledge_synthesizer.domain.errors import EmbeddingError

pytestmark = pytest.mark.unit


class _FakeEmbeddingsResource:
    def __init__(
        self, calls: list[list[str]], *, fail: bool = False, reverse: bool = False
    ) -> None:
        self._calls = calls
        self._fail = fail
        self._reverse = reverse

    def create(self, model: str, input: list[str]) -> Any:
        if self._fail:
            raise OpenAIError("boom")
        self._calls.append(list(input))
        data = [SimpleNamespace(index=i, embedding=[float(i)]) for i in range(len(input))]
        if self._reverse:
            data.reverse()
        return SimpleNamespace(data=data)


class _FakeClient:
    def __init__(self, *, fail: bool = False, reverse: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.embeddings = _FakeEmbeddingsResource(self.calls, fail=fail, reverse=reverse)


def test_embed_returns_one_vector_per_text() -> None:
    client = _FakeClient()
    vectors = OpenAIEmbeddings(client).embed(["a", "b", "c"])  # type: ignore[arg-type]
    assert vectors == [[0.0], [1.0], [2.0]]


def test_embed_batches_requests() -> None:
    client = _FakeClient()
    OpenAIEmbeddings(client, batch_size=2).embed(["a", "b", "c"])  # type: ignore[arg-type]
    assert client.calls == [["a", "b"], ["c"]]


def test_embed_orders_results_by_index() -> None:
    client = _FakeClient(reverse=True)
    vectors = OpenAIEmbeddings(client).embed(["a", "b", "c"])  # type: ignore[arg-type]
    assert vectors == [[0.0], [1.0], [2.0]]


def test_embed_empty_returns_empty() -> None:
    client = _FakeClient()
    assert OpenAIEmbeddings(client).embed([]) == []  # type: ignore[arg-type]
    assert client.calls == []


def test_embed_wraps_api_errors() -> None:
    client = _FakeClient(fail=True)
    with pytest.raises(EmbeddingError):
        OpenAIEmbeddings(client).embed(["a"])  # type: ignore[arg-type]
