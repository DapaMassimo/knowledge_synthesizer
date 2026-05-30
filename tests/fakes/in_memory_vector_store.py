"""In-memory vector store fake with cosine search."""

from __future__ import annotations

import math

from knowledge_synthesizer.domain.models import Chunk, RetrievedChunk


class InMemoryVectorStore:
    """Implements the VectorStore port. Idempotent upsert keyed by chunk_id."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._items[chunk.chunk_id] = (chunk, vector)

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        scored = [
            RetrievedChunk(chunk=chunk, score=self._cosine(vector, stored))
            for chunk, stored in self._items.values()
        ]
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:k]

    def existing_document_hashes(self) -> set[str]:
        return {chunk.document_hash for chunk, _ in self._items.values()}

    def indexed_sources(self) -> list[str]:
        return sorted({chunk.provenance.source_uri for chunk, _ in self._items.values()})

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)
