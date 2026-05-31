from pathlib import Path

import pytest

from knowledge_synthesizer.adapters.vectorstores.chroma_store import ChromaVectorStore
from knowledge_synthesizer.domain.models import Chunk, ChunkProvenance

pytestmark = pytest.mark.integration


def _chunk(
    chunk_id: str,
    text: str,
    *,
    doc_hash: str = "dh",
    page: int | None = None,
    section: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        document_hash=doc_hash,
        provenance=ChunkProvenance(source_uri="/a.pdf", page=page, section=section),
    )


def test_upsert_and_search_by_cosine_similarity() -> None:
    store = ChromaVectorStore.ephemeral(name="search_collection")
    chunks = [
        _chunk("c0", "italy rome capital", page=1, section="Geography"),
        _chunk("c1", "python programming language", page=2),
    ]
    store.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([1.0, 0.0], k=2)

    assert results[0].chunk.chunk_id == "c0"
    assert results[0].score > results[1].score
    assert results[0].chunk.text == "italy rome capital"
    assert results[0].chunk.provenance.page == 1
    assert results[0].chunk.provenance.section == "Geography"


def test_upsert_is_idempotent_on_chunk_id() -> None:
    store = ChromaVectorStore.ephemeral(name="idempotent_collection")
    chunks = [_chunk("c0", "x")]
    store.upsert(chunks, [[1.0, 0.0]])
    store.upsert(chunks, [[1.0, 0.0]])

    assert len(store.search([1.0, 0.0], k=10)) == 1
    assert store.existing_document_hashes() == {"dh"}


def test_existing_document_hashes_reports_all() -> None:
    store = ChromaVectorStore.ephemeral(name="hashes_collection")
    store.upsert(
        [_chunk("a", "x", doc_hash="h1"), _chunk("b", "y", doc_hash="h2")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    assert store.existing_document_hashes() == {"h1", "h2"}


def test_indexed_sources_lists_distinct_uris() -> None:
    store = ChromaVectorStore.ephemeral(name="sources_collection")
    store.upsert(
        [_chunk("a", "x", doc_hash="h1"), _chunk("b", "y", doc_hash="h1")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    # Both chunks share source_uri "/a.pdf" -> one distinct source.
    assert store.indexed_sources() == ["/a.pdf"]


def test_empty_upsert_is_a_noop() -> None:
    store = ChromaVectorStore.ephemeral(name="empty_collection")
    store.upsert([], [])
    assert store.existing_document_hashes() == set()


def test_persistent_recovers_from_stale_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import knowledge_synthesizer.adapters.vectorstores.chroma_store as module

    real_client = module.chromadb.PersistentClient
    calls = {"client": 0, "reset": 0}

    def flaky_client(path: str) -> object:
        calls["client"] += 1
        if calls["client"] == 1:
            raise AttributeError("'RustBindingsAPI' object has no attribute 'bindings'")
        return real_client(path=path)

    monkeypatch.setattr(module.chromadb, "PersistentClient", flaky_client)
    monkeypatch.setattr(
        module, "_reset_chroma_system_cache", lambda: calls.__setitem__("reset", calls["reset"] + 1)
    )

    store = ChromaVectorStore.persistent(tmp_path, name="recover_collection")

    assert calls["client"] == 2  # retried after the AttributeError
    assert calls["reset"] == 1  # cleared the stale system cache once
    assert store.existing_document_hashes() == set()


def test_persistent_store_round_trips(tmp_path: Path) -> None:
    ChromaVectorStore.persistent(tmp_path, name="persist_collection").upsert(
        [_chunk("a", "hello world", page=3)], [[1.0, 0.0]]
    )

    reopened = ChromaVectorStore.persistent(tmp_path, name="persist_collection")
    results = reopened.search([1.0, 0.0], k=1)

    assert results[0].chunk.chunk_id == "a"
    assert results[0].chunk.provenance.page == 3
