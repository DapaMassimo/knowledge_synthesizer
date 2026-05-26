import pytest

from knowledge_synthesizer.adapters.retrieval.query_transformers import PassthroughTransformer
from knowledge_synthesizer.adapters.retrieval.vector_retriever import VectorRetriever
from knowledge_synthesizer.application.summarization import (
    MapReduceSummarizer,
    SummarizationService,
)
from knowledge_synthesizer.domain.models import (
    Chunk,
    ChunkProvenance,
    RetrievedChunk,
    Summary,
    make_chunk_id,
)
from tests.fakes import FakeEmbeddings, FakeLLM, InMemoryVectorStore

pytestmark = pytest.mark.unit

_REDUCE_OUTPUT = (
    "Overview: Italy is a country in southern Europe.\n"
    "Key points:\n- Rome is the capital\n- Known for its cuisine"
)


def _chunk(text: str, index: int, uri: str = "/italy.md") -> Chunk:
    return Chunk(
        chunk_id=make_chunk_id("dh", index),
        text=text,
        document_hash="dh",
        provenance=ChunkProvenance(source_uri=uri),
    )


def _map_reduce_responder(system: str, prompt: str) -> str:
    if "Passage:" in prompt:
        return "NONE" if "python" in prompt else "Rome is the capital of Italy."
    return _REDUCE_OUTPUT


def test_map_reduce_produces_structured_summary() -> None:
    summarizer = MapReduceSummarizer(FakeLLM(responder=_map_reduce_responder))
    chunks = [_chunk("italy rome capital", 0), _chunk("python language", 1, uri="/py.md")]

    summary = summarizer.summarize("Italy", chunks)

    assert summary.topic == "Italy"
    assert summary.overview == "Italy is a country in southern Europe."
    assert summary.key_points == ["Rome is the capital", "Known for its cuisine"]
    # The irrelevant 'python' chunk mapped to NONE and is not cited.
    assert [c.source_uri for c in summary.citations] == ["/italy.md"]


def test_map_reduce_handles_all_irrelevant_chunks() -> None:
    summarizer = MapReduceSummarizer(FakeLLM(response="NONE"))
    summary = summarizer.summarize("Italy", [_chunk("unrelated", 0)])
    assert summary.key_points == []
    assert summary.citations == []
    assert "do not cover" in summary.overview


def test_parse_summary_falls_back_to_plain_text() -> None:
    summarizer = MapReduceSummarizer(
        FakeLLM(responder=lambda s, p: "Rome." if "Passage:" in p else "Just an overview.")
    )
    summary = summarizer.summarize("Italy", [_chunk("rome", 0)])
    assert summary.overview == "Just an overview."
    assert summary.key_points == []


class _StubSummarizer:
    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []

    def summarize(self, topic: str, chunks: list[Chunk]) -> Summary:
        self.seen.append((topic, len(chunks)))
        return Summary(topic=topic, overview="ok", key_points=["p"])


class _StubRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        return [RetrievedChunk(chunk=chunk, score=1.0) for chunk in self._chunks[:k]]


async def test_service_gathers_chunks_and_delegates() -> None:
    summarizer = _StubSummarizer()
    service = SummarizationService(_StubRetriever([_chunk("a", 0), _chunk("b", 1)]), summarizer)

    summary = await service.summarize("Italy")

    assert summary.overview == "ok"
    assert summarizer.seen == [("Italy", 2)]


async def test_service_with_no_chunks_returns_fallback() -> None:
    service = SummarizationService(_StubRetriever([]), _StubSummarizer())
    summary = await service.summarize("Italy")
    assert "No sources are indexed" in summary.overview


async def test_end_to_end_summary_with_real_retriever() -> None:
    embeddings = FakeEmbeddings(dim=64)
    store = InMemoryVectorStore()
    chunks = [_chunk("italy rome capital city", 0)]
    store.upsert(chunks, embeddings.embed([c.text for c in chunks]))
    retriever = VectorRetriever(embeddings, store, PassthroughTransformer())
    service = SummarizationService(
        retriever, MapReduceSummarizer(FakeLLM(responder=_map_reduce_responder)), k=5
    )

    summary = await service.summarize("Italy")

    assert summary.key_points
    assert summary.citations[0].source_uri == "/italy.md"
