import pytest

from knowledge_synthesizer.application.indexing import IndexingService
from knowledge_synthesizer.domain.models import (
    Chunk,
    ChunkProvenance,
    FileSource,
    ParsedDocument,
    RawDocument,
    Source,
    make_chunk_id,
)
from tests.fakes import FakeEmbeddings, InMemoryVectorStore

pytestmark = pytest.mark.unit


class _StubLoader:
    """Returns one RawDocument per source; content is keyed so hashes are predictable."""

    def __init__(self, content_by_uri: dict[str, bytes]) -> None:
        self._content_by_uri = content_by_uri

    async def load(self, source: Source) -> list[RawDocument]:
        return [
            RawDocument(
                content=self._content_by_uri[source.uri],
                mime_type="text/plain",
                source=source,
            )
        ]


class _StubParser:
    def parse(self, raw: RawDocument) -> ParsedDocument:
        return ParsedDocument(
            source=raw.source,
            content_hash=raw.content_hash,
            markdown=raw.content.decode("utf-8"),
        )


class _StubChunker:
    """Splits the markdown into one chunk per line."""

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        lines = [line for line in doc.markdown.splitlines() if line.strip()]
        return [
            Chunk(
                chunk_id=make_chunk_id(doc.content_hash, index),
                text=line,
                document_hash=doc.content_hash,
                provenance=ChunkProvenance(source_uri=doc.source.uri),
            )
            for index, line in enumerate(lines)
        ]


def _service(content_by_uri: dict[str, bytes], store: InMemoryVectorStore) -> IndexingService:
    return IndexingService(
        loader=_StubLoader(content_by_uri),
        parser=_StubParser(),
        chunker=_StubChunker(),
        embeddings=FakeEmbeddings(),
        store=store,
    )


async def test_indexes_documents_and_chunks() -> None:
    store = InMemoryVectorStore()
    content = {"/a.txt": b"alpha\nbeta", "/b.txt": b"gamma"}
    service = _service(content, store)

    report = await service.index([FileSource(path="/a.txt"), FileSource(path="/b.txt")])

    assert report.documents_indexed == 2
    assert report.documents_skipped == 0
    assert report.chunks_indexed == 3  # 2 lines + 1 line
    assert len(store) == 3


async def test_reindexing_is_idempotent() -> None:
    store = InMemoryVectorStore()
    content = {"/a.txt": b"alpha\nbeta"}
    service = _service(content, store)
    sources = [FileSource(path="/a.txt")]

    first = await service.index(sources)
    second = await service.index(sources)

    assert first.documents_indexed == 1
    assert second.documents_indexed == 0
    assert second.documents_skipped == 1
    assert len(store) == 2  # unchanged


async def test_duplicate_content_within_one_run_is_skipped() -> None:
    store = InMemoryVectorStore()
    # Two different sources, identical bytes -> same content_hash.
    content = {"/a.txt": b"same content", "/b.txt": b"same content"}
    service = _service(content, store)

    report = await service.index([FileSource(path="/a.txt"), FileSource(path="/b.txt")])

    assert report.documents_indexed == 1
    assert report.documents_skipped == 1
