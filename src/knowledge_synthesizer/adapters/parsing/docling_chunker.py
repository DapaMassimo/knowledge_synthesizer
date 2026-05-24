"""Split a parsed document into provenance-carrying chunks using Docling's chunker.

Implements the Chunker port. The native DoclingDocument is retrieved from the shared
cache (populated by DoclingParser) keyed by ``content_hash``.
"""

from __future__ import annotations

from docling.chunking import (  # type: ignore[attr-defined]
    BaseChunk,
    BaseChunker,
    HybridChunker,
)

from knowledge_synthesizer.adapters.parsing.docling_cache import DoclingDocumentCache
from knowledge_synthesizer.domain.errors import ChunkingError
from knowledge_synthesizer.domain.models import (
    Chunk,
    ChunkProvenance,
    ParsedDocument,
    make_chunk_id,
)


class DoclingChunker:
    """Runs a Docling chunker (HybridChunker by default) and maps to domain Chunks."""

    def __init__(self, cache: DoclingDocumentCache, chunker: BaseChunker | None = None) -> None:
        self._cache = cache
        self._chunker: BaseChunker = chunker if chunker is not None else HybridChunker()

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        native = self._cache.get(doc.content_hash)
        if native is None:
            raise ChunkingError(
                f"no cached DoclingDocument for {doc.source.uri!r}; parse the document first"
            )
        chunks: list[Chunk] = []
        for index, docling_chunk in enumerate(self._chunker.chunk(native)):
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.content_hash, index),
                    text=self._chunker.contextualize(docling_chunk),
                    document_hash=doc.content_hash,
                    provenance=ChunkProvenance(
                        source_uri=doc.source.uri,
                        page=self._first_page_no(docling_chunk),
                        section=self._section(docling_chunk),
                    ),
                )
            )
        return chunks

    @staticmethod
    def _section(docling_chunk: BaseChunk) -> str | None:
        headings = getattr(docling_chunk.meta, "headings", None)
        if headings:
            return " > ".join(str(heading) for heading in headings)
        return None

    @staticmethod
    def _first_page_no(docling_chunk: BaseChunk) -> int | None:
        for item in getattr(docling_chunk.meta, "doc_items", []) or []:
            for prov in getattr(item, "prov", []) or []:
                page_no = getattr(prov, "page_no", None)
                if isinstance(page_no, int):
                    return page_no
        return None
