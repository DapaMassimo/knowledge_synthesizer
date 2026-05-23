"""Ports: structural interfaces (Protocols) the application depends on. No frameworks."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from knowledge_synthesizer.domain.models import (
    Chunk,
    ParsedDocument,
    RawDocument,
    RetrievedChunk,
    Source,
)


@runtime_checkable
class SourceLoader(Protocol):
    """Load the raw bytes for a source (file or web)."""

    async def load(self, source: Source) -> list[RawDocument]: ...


@runtime_checkable
class DocumentParser(Protocol):
    """Parse raw bytes into normalized text/markdown with structure."""

    def parse(self, raw: RawDocument) -> ParsedDocument: ...


@runtime_checkable
class Chunker(Protocol):
    """Split a parsed document into provenance-carrying chunks."""

    def chunk(self, doc: ParsedDocument) -> list[Chunk]: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    """Turn texts into dense vectors. The adapter owns the model and dimensionality."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persist chunk vectors and search them by cosine similarity."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]: ...

    def existing_document_hashes(self) -> set[str]:
        """Document hashes already stored — enables idempotent incremental indexing."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Retrieve the top-k chunks relevant to a natural-language query."""

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Single-turn completion given a system prompt and a user prompt."""

    def complete(self, system: str, prompt: str) -> str: ...


@runtime_checkable
class QueryTransformer(Protocol):
    """Turn a raw user question into one or more search queries."""

    def transform(self, question: str) -> list[str]: ...
