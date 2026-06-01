"""Chroma vector store adapter. Implements the VectorStore port (cosine, precomputed vectors)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb import Collection

from knowledge_synthesizer.domain.errors import VectorStoreError
from knowledge_synthesizer.domain.models import Chunk, ChunkProvenance, RetrievedChunk

_COSINE_SPACE = {"hnsw:space": "cosine"}
_DEFAULT_COLLECTION = "knowledge"


class ChromaVectorStore:
    """Stores chunk vectors in a Chroma collection and searches by cosine similarity."""

    def __init__(self, collection: Collection) -> None:
        self._collection = collection

    @classmethod
    def ephemeral(cls, name: str = _DEFAULT_COLLECTION) -> ChromaVectorStore:
        client = chromadb.EphemeralClient()
        return cls(client.get_or_create_collection(name=name, metadata=_COSINE_SPACE))

    @classmethod
    def persistent(cls, path: Path | str, name: str = _DEFAULT_COLLECTION) -> ChromaVectorStore:
        path_str = str(path)
        try:
            client = chromadb.PersistentClient(path=path_str)
        except (AttributeError, KeyError):
            # Corrupted in-process Chroma state — typically caused by deleting the persist
            # dir while the app held it open. Drop the cached systems and retry once.
            _reset_chroma_system_cache()
            try:
                client = chromadb.PersistentClient(path=path_str)
            except (AttributeError, KeyError) as exc:
                raise VectorStoreError(
                    f"Chroma is in a corrupted in-process state (often from deleting "
                    f"{path_str!r} while the app was running). Restart the app to recover."
                ) from exc
        return cls(client.get_or_create_collection(name=name, metadata=_COSINE_SPACE))

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                embeddings=cast("Any", vectors),
                documents=[chunk.text for chunk in chunks],
                metadatas=[self._metadata(chunk) for chunk in chunks],
            )
        except (chromadb.errors.ChromaError, ValueError) as exc:
            raise VectorStoreError(f"chroma upsert failed: {exc}") from exc

    def search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        try:
            raw = self._collection.query(
                query_embeddings=cast("Any", [vector]),
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except (chromadb.errors.ChromaError, ValueError) as exc:
            raise VectorStoreError(f"chroma search failed: {exc}") from exc

        result = cast("dict[str, Any]", raw)
        ids = _first(result.get("ids"))
        documents = _first(result.get("documents"))
        metadatas = _first(result.get("metadatas"))
        distances = _first(result.get("distances"))
        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            chunk = _to_chunk(str(chunk_id), str(text), metadata or {})
            retrieved.append(RetrievedChunk(chunk=chunk, score=1.0 - float(distance)))
        return retrieved

    def existing_document_hashes(self) -> set[str]:
        return {
            str(metadata["document_hash"])
            for metadata in self._all_metadatas()
            if metadata.get("document_hash") is not None
        }

    def indexed_sources(self) -> list[str]:
        return sorted(
            {
                str(metadata["source_uri"])
                for metadata in self._all_metadatas()
                if metadata.get("source_uri") is not None
            }
        )

    def _all_metadatas(self) -> list[dict[str, Any]]:
        try:
            raw = self._collection.get(include=["metadatas"])
        except (chromadb.errors.ChromaError, ValueError) as exc:
            raise VectorStoreError(f"chroma get failed: {exc}") from exc
        result = cast("dict[str, Any]", raw)
        return [metadata for metadata in (result.get("metadatas") or []) if metadata]

    @staticmethod
    def _metadata(chunk: Chunk) -> dict[str, str | int]:
        metadata: dict[str, str | int] = {
            "document_hash": chunk.document_hash,
            "source_uri": chunk.provenance.source_uri,
        }
        if chunk.provenance.page is not None:
            metadata["page"] = chunk.provenance.page
        if chunk.provenance.section is not None:
            metadata["section"] = chunk.provenance.section
        return metadata


def _reset_chroma_system_cache() -> None:
    """Clear Chroma's cached systems (drops references without calling their buggy stop())."""
    from chromadb.api.shared_system_client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


def _first(values: Any) -> list[Any]:
    if not values:
        return []
    return list(values[0]) if values[0] else []


def _to_chunk(chunk_id: str, text: str, metadata: dict[str, Any]) -> Chunk:
    page = metadata.get("page")
    section = metadata.get("section")
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        document_hash=str(metadata.get("document_hash", "")),
        provenance=ChunkProvenance(
            source_uri=str(metadata.get("source_uri", "")),
            page=int(page) if isinstance(page, int | float) else None,
            section=str(section) if section is not None else None,
        ),
    )
