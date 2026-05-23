"""Domain entities and value objects. Immutable Pydantic models, no framework deps."""

from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

_FROZEN = ConfigDict(frozen=True)


class WebSource(BaseModel):
    """A web page identified by URL."""

    model_config = _FROZEN

    kind: Literal["web"] = "web"
    url: str

    @property
    def uri(self) -> str:
        return self.url


class FileSource(BaseModel):
    """A local file identified by filesystem path."""

    model_config = _FROZEN

    kind: Literal["file"] = "file"
    path: str

    @property
    def uri(self) -> str:
        return self.path


Source = Annotated[WebSource | FileSource, Field(discriminator="kind")]
"""Discriminated union of source kinds. Use a TypeAdapter to parse from raw data."""


class RawDocument(BaseModel):
    """Unparsed bytes loaded from a source, with a content hash for idempotent indexing."""

    model_config = _FROZEN

    content: bytes
    mime_type: str
    source: Source

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


class ParsedDocument(BaseModel):
    """Normalized text/markdown plus light structural metadata, ready for chunking."""

    model_config = _FROZEN

    source: Source
    content_hash: str
    markdown: str
    title: str | None = None
    page_count: int | None = None


class ChunkProvenance(BaseModel):
    """Where a chunk came from — required for citations."""

    model_config = _FROZEN

    source_uri: str
    page: int | None = None
    section: str | None = None


class Chunk(BaseModel):
    """A retrievable unit of text with provenance and the parent document's hash."""

    model_config = _FROZEN

    chunk_id: str
    text: str
    document_hash: str
    provenance: ChunkProvenance


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval together with its similarity score."""

    model_config = _FROZEN

    chunk: Chunk
    score: float


class Citation(BaseModel):
    """A pointer back to a source location supporting an answer or summary."""

    model_config = _FROZEN

    source_uri: str
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    snippet: str | None = None

    @classmethod
    def from_chunk(cls, chunk: Chunk, snippet: str | None = None) -> Citation:
        return cls(
            source_uri=chunk.provenance.source_uri,
            page=chunk.provenance.page,
            section=chunk.provenance.section,
            chunk_id=chunk.chunk_id,
            snippet=snippet,
        )


class Answer(BaseModel):
    """A grounded answer to a question with supporting citations."""

    model_config = _FROZEN

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)


class Summary(BaseModel):
    """A structured topic summary: an overview plus key points, with citations."""

    model_config = _FROZEN

    topic: str
    overview: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


def make_chunk_id(document_hash: str, index: int) -> str:
    """Deterministic chunk id so re-indexing identical content overwrites idempotently."""
    return hashlib.sha256(f"{document_hash}:{index}".encode()).hexdigest()[:32]
