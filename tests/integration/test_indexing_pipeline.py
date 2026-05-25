from pathlib import Path

import pytest
from docling.chunking import HierarchicalChunker

from knowledge_synthesizer.adapters.loaders.file_loader import FileLoader
from knowledge_synthesizer.adapters.parsing.docling_cache import DoclingDocumentCache
from knowledge_synthesizer.adapters.parsing.docling_chunker import DoclingChunker
from knowledge_synthesizer.adapters.parsing.docling_parser import DoclingParser
from knowledge_synthesizer.application.indexing import IndexingService
from knowledge_synthesizer.domain.models import FileSource
from tests.fakes import FakeEmbeddings, InMemoryVectorStore

pytestmark = pytest.mark.integration

_MARKDOWN = """# Italy

Rome is the capital of Italy with a long history.

## Cuisine

Italian cuisine includes pasta and pizza.
"""


def _service(store: InMemoryVectorStore) -> IndexingService:
    cache = DoclingDocumentCache()
    return IndexingService(
        loader=FileLoader(),
        parser=DoclingParser(cache=cache),
        chunker=DoclingChunker(cache=cache, chunker=HierarchicalChunker()),
        embeddings=FakeEmbeddings(),
        store=store,
    )


async def test_real_pipeline_indexes_a_markdown_file(tmp_path: Path) -> None:
    path = tmp_path / "italy.md"
    path.write_text(_MARKDOWN, encoding="utf-8")
    store = InMemoryVectorStore()

    report = await _service(store).index([FileSource(path=str(path))])

    assert report.documents_indexed == 1
    assert report.chunks_indexed > 0
    assert len(store) == report.chunks_indexed
    assert store.existing_document_hashes()


async def test_real_pipeline_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "italy.md"
    path.write_text(_MARKDOWN, encoding="utf-8")
    store = InMemoryVectorStore()
    service = _service(store)
    sources = [FileSource(path=str(path))]

    await service.index(sources)
    size_after_first = len(store)
    second = await service.index(sources)

    assert second.documents_indexed == 0
    assert second.documents_skipped == 1
    assert len(store) == size_after_first
