"""Indexing use case: load -> parse -> chunk -> embed -> upsert, idempotently."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict

from knowledge_synthesizer.domain.models import RawDocument, Source
from knowledge_synthesizer.domain.ports import (
    Chunker,
    DocumentParser,
    EmbeddingModel,
    SourceLoader,
    VectorStore,
)


class IndexReport(BaseModel):
    """Outcome of an indexing run."""

    model_config = ConfigDict(frozen=True)

    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_indexed: int = 0


class IndexingService:
    """Orchestrates the indexing pipeline. Depends only on ports.

    Idempotent: a document whose ``content_hash`` is already in the store is skipped, so
    re-indexing a source set only processes new or changed documents.
    """

    def __init__(
        self,
        loader: SourceLoader,
        parser: DocumentParser,
        chunker: Chunker,
        embeddings: EmbeddingModel,
        store: VectorStore,
    ) -> None:
        self._loader = loader
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store

    async def index(self, sources: list[Source]) -> IndexReport:
        seen = set(self._store.existing_document_hashes())
        documents_indexed = 0
        documents_skipped = 0
        chunks_indexed = 0

        for source in sources:
            for raw in await self._loader.load(source):
                if raw.content_hash in seen:
                    documents_skipped += 1
                    continue
                # Docling parsing is CPU-bound; keep it off the event loop.
                chunks_indexed += await asyncio.to_thread(self._process_document, raw)
                seen.add(raw.content_hash)
                documents_indexed += 1

        return IndexReport(
            documents_indexed=documents_indexed,
            documents_skipped=documents_skipped,
            chunks_indexed=chunks_indexed,
        )

    def _process_document(self, raw: RawDocument) -> int:
        parsed = self._parser.parse(raw)
        chunks = self._chunker.chunk(parsed)
        if chunks:
            vectors = self._embeddings.embed([chunk.text for chunk in chunks])
            self._store.upsert(chunks, vectors)
        return len(chunks)
