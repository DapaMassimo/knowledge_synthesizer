"""Framework-free helpers to index the eval corpus and build ragas-shaped records."""

from __future__ import annotations

import hashlib

from knowledge_synthesizer.application.qa import QAService
from knowledge_synthesizer.domain.models import Chunk, ChunkProvenance, make_chunk_id
from knowledge_synthesizer.domain.ports import EmbeddingModel, Retriever, VectorStore
from tests.eval.dataset import EvalSample


def index_corpus(
    embeddings: EmbeddingModel,
    store: VectorStore,
    corpus: list[tuple[str, str]],
) -> None:
    """Index a corpus of (source_uri, text) as one chunk per document."""
    chunks: list[Chunk] = []
    for uri, text in corpus:
        document_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(document_hash, 0),
                text=text,
                document_hash=document_hash,
                provenance=ChunkProvenance(source_uri=uri),
            )
        )
    store.upsert(chunks, embeddings.embed([chunk.text for chunk in chunks]))


async def build_records(
    qa: QAService,
    retriever: Retriever,
    samples: list[EvalSample],
    k: int = 4,
) -> list[dict[str, object]]:
    """Produce ragas evaluation records: user_input, response, retrieved_contexts, reference."""
    records: list[dict[str, object]] = []
    for sample in samples:
        answer = await qa.ask(sample.question)
        contexts = [hit.chunk.text for hit in retriever.retrieve(sample.question, k)]
        records.append(
            {
                "user_input": sample.question,
                "response": answer.text,
                "retrieved_contexts": contexts,
                "reference": sample.ground_truth,
            }
        )
    return records
