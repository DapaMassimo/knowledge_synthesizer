import hashlib

import pytest
from pydantic import TypeAdapter, ValidationError

from knowledge_synthesizer.domain.models import (
    Answer,
    Chunk,
    ChunkProvenance,
    Citation,
    FileSource,
    RawDocument,
    RetrievedChunk,
    Source,
    Summary,
    WebSource,
    make_chunk_id,
)

pytestmark = pytest.mark.unit

_SOURCE_ADAPTER: TypeAdapter[Source] = TypeAdapter(Source)


def test_web_and_file_source_uri() -> None:
    assert WebSource(url="https://example.com").uri == "https://example.com"
    assert FileSource(path="/tmp/a.pdf").uri == "/tmp/a.pdf"


def test_source_discriminated_union_parses_by_kind() -> None:
    web = _SOURCE_ADAPTER.validate_python({"kind": "web", "url": "https://x.org"})
    file = _SOURCE_ADAPTER.validate_python({"kind": "file", "path": "/d/f.txt"})
    assert isinstance(web, WebSource)
    assert isinstance(file, FileSource)


def test_source_discriminator_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        _SOURCE_ADAPTER.validate_python({"kind": "ftp", "url": "ftp://x"})


def test_raw_document_content_hash_is_sha256() -> None:
    raw = RawDocument(content=b"hello", mime_type="text/plain", source=FileSource(path="/a.txt"))
    assert raw.content_hash == hashlib.sha256(b"hello").hexdigest()


def test_raw_document_content_hash_is_deterministic() -> None:
    src = WebSource(url="https://x")
    a = RawDocument(content=b"same", mime_type="text/plain", source=src)
    b = RawDocument(content=b"same", mime_type="text/plain", source=src)
    assert a.content_hash == b.content_hash


def test_models_are_frozen() -> None:
    raw = RawDocument(content=b"x", mime_type="text/plain", source=FileSource(path="/a"))
    with pytest.raises(ValidationError):
        raw.mime_type = "text/html"  # type: ignore[misc]


def test_make_chunk_id_is_deterministic_and_position_sensitive() -> None:
    assert make_chunk_id("dochash", 0) == make_chunk_id("dochash", 0)
    assert make_chunk_id("dochash", 0) != make_chunk_id("dochash", 1)
    assert make_chunk_id("a", 0) != make_chunk_id("b", 0)


def test_citation_from_chunk_carries_provenance() -> None:
    chunk = Chunk(
        chunk_id="c1",
        text="Rome is the capital of Italy.",
        document_hash="dh",
        provenance=ChunkProvenance(source_uri="/a.pdf", page=3, section="Intro"),
    )
    citation = Citation.from_chunk(chunk, snippet="Rome is the capital")
    assert citation.source_uri == "/a.pdf"
    assert citation.page == 3
    assert citation.section == "Intro"
    assert citation.chunk_id == "c1"
    assert citation.snippet == "Rome is the capital"


def test_retrieved_chunk_and_answer_and_summary_construct() -> None:
    chunk = Chunk(
        chunk_id="c1",
        text="text",
        document_hash="dh",
        provenance=ChunkProvenance(source_uri="/a.pdf"),
    )
    rc = RetrievedChunk(chunk=chunk, score=0.87)
    assert rc.score == pytest.approx(0.87)

    answer = Answer(question="Q?", text="A.", citations=[Citation.from_chunk(chunk)])
    assert answer.citations[0].chunk_id == "c1"

    summary = Summary(topic="Italy", overview="...", key_points=["Rome is capital"])
    assert summary.key_points == ["Rome is capital"]
    assert summary.citations == []
