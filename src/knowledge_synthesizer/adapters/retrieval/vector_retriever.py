"""Vector retriever: transform the query, embed each search query, union the hits.

Implements the Retriever port. Composes an EmbeddingModel, a VectorStore, and a
QueryTransformer (all injected and swappable). Similarity-only (cosine top-k); the
transformer decides whether that's passthrough, multi-query, or HyDE.
"""

from __future__ import annotations

from knowledge_synthesizer.domain.models import RetrievedChunk
from knowledge_synthesizer.domain.ports import (
    EmbeddingModel,
    QueryTransformer,
    VectorStore,
)


class VectorRetriever:
    def __init__(
        self,
        embeddings: EmbeddingModel,
        store: VectorStore,
        transformer: QueryTransformer,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._transformer = transformer

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        best: dict[str, RetrievedChunk] = {}
        for search_query in self._transformer.transform(query):
            vectors = self._embeddings.embed([search_query])
            if not vectors:
                continue
            for hit in self._store.search(vectors[0], k):
                current = best.get(hit.chunk.chunk_id)
                if current is None or hit.score > current.score:
                    best[hit.chunk.chunk_id] = hit
        ranked = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
        return ranked[:k]
