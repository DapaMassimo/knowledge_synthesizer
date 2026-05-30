import pytest

from knowledge_synthesizer.domain.models import Chunk, ChunkProvenance, make_chunk_id
from tests.fakes import FakeEmbeddings, FakeLLM, InMemoryVectorStore

pytestmark = pytest.mark.unit


def _chunk(text: str, index: int, doc_hash: str = "dh") -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id(doc_hash, index),
        text=text,
        document_hash=doc_hash,
        provenance=ChunkProvenance(source_uri="/a.pdf", page=index),
    )


def test_fake_embeddings_are_deterministic_and_right_shape() -> None:
    emb = FakeEmbeddings(dim=16)
    [v1] = emb.embed(["hello world"])
    [v2] = emb.embed(["hello world"])
    assert v1 == v2
    assert len(v1) == 16


def test_fake_embeddings_empty_text_is_zero_vector() -> None:
    emb = FakeEmbeddings(dim=8)
    [v] = emb.embed([""])
    assert v == [0.0] * 8


def test_fake_llm_modes() -> None:
    assert FakeLLM(response="canned").complete("s", "p") == "canned"
    assert FakeLLM().complete("s", "echo me") == "echo me"
    llm = FakeLLM(responder=lambda system, prompt: prompt.upper())
    assert llm.complete("s", "hi") == "HI"
    assert llm.calls == [("s", "hi")]


def test_vector_store_upsert_is_idempotent_and_searches_by_similarity() -> None:
    emb = FakeEmbeddings(dim=32)
    store = InMemoryVectorStore()
    chunks = [
        _chunk("italy rome capital", 0),
        _chunk("python programming language", 1),
    ]
    store.upsert(chunks, emb.embed([c.text for c in chunks]))
    store.upsert(chunks, emb.embed([c.text for c in chunks]))  # repeat -> no duplicates
    assert len(store) == 2

    [query] = emb.embed(["what is the capital of italy"])
    results = store.search(query, k=2)
    assert results[0].chunk.text == "italy rome capital"
    assert results[0].score >= results[1].score


def test_vector_store_reports_existing_document_hashes() -> None:
    emb = FakeEmbeddings()
    store = InMemoryVectorStore()
    chunks = [_chunk("a", 0, "hash-a"), _chunk("b", 0, "hash-b")]
    store.upsert(chunks, emb.embed([c.text for c in chunks]))
    assert store.existing_document_hashes() == {"hash-a", "hash-b"}
    assert store.indexed_sources() == ["/a.pdf"]  # both chunks share the same source_uri
