import pytest

from knowledge_synthesizer.adapters.retrieval.query_transformers import (
    MultiQueryTransformer,
    PassthroughTransformer,
)
from knowledge_synthesizer.adapters.retrieval.vector_retriever import VectorRetriever
from knowledge_synthesizer.domain.models import Chunk, ChunkProvenance, make_chunk_id
from tests.fakes import FakeEmbeddings, FakeLLM, InMemoryVectorStore

pytestmark = pytest.mark.unit


def _chunk(text: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id("dh", index),
        text=text,
        document_hash="dh",
        provenance=ChunkProvenance(source_uri="/a.md"),
    )


def _populated_store(embeddings: FakeEmbeddings) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    chunks = [
        _chunk("italy rome capital", 0),
        _chunk("python programming language", 1),
        _chunk("france paris capital", 2),
    ]
    store.upsert(chunks, embeddings.embed([c.text for c in chunks]))
    return store


def test_passthrough_retrieval_ranks_by_similarity() -> None:
    embeddings = FakeEmbeddings(dim=64)
    retriever = VectorRetriever(embeddings, _populated_store(embeddings), PassthroughTransformer())

    results = retriever.retrieve("capital of italy rome", k=2)

    assert len(results) == 2
    assert results[0].chunk.text == "italy rome capital"
    assert results[0].score >= results[1].score


def test_multiquery_unions_and_dedupes_by_chunk_id() -> None:
    embeddings = FakeEmbeddings(dim=64)
    store = _populated_store(embeddings)
    # Two reformulations that both surface the italy chunk.
    llm = FakeLLM(response="rome capital\nitaly")
    retriever = VectorRetriever(embeddings, store, MultiQueryTransformer(llm))

    results = retriever.retrieve("italy rome", k=3)

    chunk_ids = [r.chunk.chunk_id for r in results]
    assert len(chunk_ids) == len(set(chunk_ids))  # no duplicates across queries
    assert results[0].chunk.text == "italy rome capital"


def test_retrieve_respects_k() -> None:
    embeddings = FakeEmbeddings(dim=64)
    retriever = VectorRetriever(embeddings, _populated_store(embeddings), PassthroughTransformer())
    assert len(retriever.retrieve("capital", k=1)) == 1
