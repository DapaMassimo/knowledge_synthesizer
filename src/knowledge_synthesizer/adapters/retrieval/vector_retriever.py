"""Vector retriever: transform the query, embed each search query, union the hits.

Implements the Retriever port. Composes an EmbeddingModel, a VectorStore, and a
QueryTransformer (all injected and swappable). Similarity-only (cosine top-k); the
transformer decides whether that's passthrough, multi-query, or HyDE.
"""

from __future__ import annotations

import logging

from knowledge_synthesizer.domain.models import RetrievedChunk
from knowledge_synthesizer.domain.ports import (
    EmbeddingModel,
    QueryTransformer,
    VectorStore,
)

logger = logging.getLogger(__name__)


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
        search_queries = self._transformer.transform(query)
        logger.info(
            "Retrieve %r (k=%d) via %d search query(ies): %s",
            query,
            k,
            len(search_queries),
            search_queries,
        )
        best: dict[str, RetrievedChunk] = {}
        for search_query in search_queries:
            vectors = self._embeddings.embed([search_query])
            if not vectors:
                logger.warning("No embedding produced for search query %r", search_query)
                continue
            hits = self._store.search(vectors[0], k)
            logger.info(
                "  search %r → %d hit(s) [top score %.3f]",
                search_query,
                len(hits),
                hits[0].score if hits else 0.0,
            )
            for hit in hits:
                current = best.get(hit.chunk.chunk_id)
                if current is None or hit.score > current.score:
                    best[hit.chunk.chunk_id] = hit
        ranked = sorted(best.values(), key=lambda hit: hit.score, reverse=True)
        logger.info(
            "Retrieve %r → %d unique chunk(s) after union/dedup, returning top %d",
            query,
            len(best),
            min(k, len(ranked)),
        )
        for rank, hit in enumerate(ranked[:k], start=1):
            prov = hit.chunk.provenance
            page = f", p.{prov.page}" if prov.page is not None else ""
            logger.debug(
                "    #%d score %.3f — %s%s: %.80s",
                rank,
                hit.score,
                prov.source_uri,
                page,
                hit.chunk.text.replace("\n", " "),
            )
        return ranked[:k]
