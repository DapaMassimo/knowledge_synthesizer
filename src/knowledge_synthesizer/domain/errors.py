"""Typed domain exceptions. Adapters translate library errors into these at the boundary."""

from __future__ import annotations


class KnowledgeSynthesizerError(Exception):
    """Base class for all domain errors."""


class SourceLoadError(KnowledgeSynthesizerError):
    """A source could not be loaded (missing file, network failure, bad URL)."""


class ParsingError(KnowledgeSynthesizerError):
    """A raw document could not be parsed into a normalized document."""


class ChunkingError(KnowledgeSynthesizerError):
    """A parsed document could not be split into chunks."""


class EmbeddingError(KnowledgeSynthesizerError):
    """Embedding computation failed."""


class VectorStoreError(KnowledgeSynthesizerError):
    """A vector store operation (upsert/search) failed."""


class RetrievalError(KnowledgeSynthesizerError):
    """Retrieval failed."""


class LLMError(KnowledgeSynthesizerError):
    """An LLM completion failed."""


class ConfigurationError(KnowledgeSynthesizerError):
    """The application was misconfigured (missing key, unknown backend)."""
