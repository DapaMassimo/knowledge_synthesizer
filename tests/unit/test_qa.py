import pytest

from knowledge_synthesizer.adapters.retrieval.query_transformers import PassthroughTransformer
from knowledge_synthesizer.adapters.retrieval.vector_retriever import VectorRetriever
from knowledge_synthesizer.application.qa import QAService
from knowledge_synthesizer.domain.models import (
    Chunk,
    ChunkProvenance,
    RetrievedChunk,
    make_chunk_id,
)
from tests.fakes import FakeEmbeddings, FakeLLM, InMemoryVectorStore

pytestmark = pytest.mark.unit


def _retrieved(
    text: str, index: int, *, page: int | None = None, uri: str = "/a.pdf"
) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=make_chunk_id("dh", index),
        text=text,
        document_hash="dh",
        provenance=ChunkProvenance(source_uri=uri, page=page),
    )
    return RetrievedChunk(chunk=chunk, score=1.0 - index * 0.1)


class _StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return self._chunks[:k]


async def test_answer_includes_only_cited_sources() -> None:
    retrieved = [
        _retrieved("Rome is the capital of Italy.", 0, page=1, uri="/italy.pdf"),
        _retrieved("Paris is the capital of France.", 1, page=2, uri="/france.pdf"),
    ]
    llm = FakeLLM(response="Rome is the capital of Italy [1].")
    answer = await QAService(_StubRetriever(retrieved), llm).ask("capital of italy?")

    assert answer.text == "Rome is the capital of Italy [1]."
    assert len(answer.citations) == 1
    assert answer.citations[0].source_uri == "/italy.pdf"
    assert answer.citations[0].page == 1
    assert answer.citations[0].snippet == "Rome is the capital of Italy."


async def test_prompt_contains_numbered_context() -> None:
    retrieved = [_retrieved("Rome is the capital.", 0, page=3)]
    llm = FakeLLM(response="answer [1]")
    await QAService(_StubRetriever(retrieved), llm).ask("q?")

    _system, prompt = llm.calls[0]
    assert "[1]" in prompt
    assert "page 3" in prompt
    assert "Rome is the capital." in prompt


async def test_answer_without_explicit_citations_includes_all() -> None:
    retrieved = [_retrieved("a", 0), _retrieved("b", 1)]
    llm = FakeLLM(response="No bracket references here.")
    answer = await QAService(_StubRetriever(retrieved), llm).ask("q?")
    assert len(answer.citations) == 2


async def test_out_of_range_citation_falls_back_to_all() -> None:
    retrieved = [_retrieved("a", 0), _retrieved("b", 1)]
    llm = FakeLLM(response="see [9]")
    answer = await QAService(_StubRetriever(retrieved), llm).ask("q?")
    assert len(answer.citations) == 2


async def test_no_results_returns_dont_know_without_calling_llm() -> None:
    llm = FakeLLM(response="should not be used")
    answer = await QAService(_StubRetriever([]), llm).ask("q?")
    assert "don't have enough information" in answer.text
    assert answer.citations == []
    assert llm.calls == []


async def test_end_to_end_with_real_retriever() -> None:
    embeddings = FakeEmbeddings(dim=64)
    store = InMemoryVectorStore()
    chunks = [
        Chunk(
            chunk_id=make_chunk_id("dh", 0),
            text="italy rome capital",
            document_hash="dh",
            provenance=ChunkProvenance(source_uri="/italy.md", page=1),
        ),
        Chunk(
            chunk_id=make_chunk_id("dh", 1),
            text="python programming language",
            document_hash="dh",
            provenance=ChunkProvenance(source_uri="/py.md"),
        ),
    ]
    store.upsert(chunks, embeddings.embed([c.text for c in chunks]))
    retriever = VectorRetriever(embeddings, store, PassthroughTransformer())
    llm = FakeLLM(responder=lambda system, prompt: "Rome is the capital of Italy [1].")

    answer = await QAService(retriever, llm, k=2).ask("what is the capital of italy")

    assert answer.citations[0].source_uri == "/italy.md"
    assert answer.citations[0].page == 1
