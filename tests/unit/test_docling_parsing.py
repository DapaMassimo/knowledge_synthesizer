import io
from pathlib import Path

import pytest
from docling.chunking import HierarchicalChunker
from docling.datamodel.document import ConversionResult
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream

from knowledge_synthesizer.adapters.parsing.docling_cache import DoclingDocumentCache
from knowledge_synthesizer.adapters.parsing.docling_chunker import DoclingChunker
from knowledge_synthesizer.adapters.parsing.docling_parser import DoclingParser
from knowledge_synthesizer.domain.errors import ChunkingError, ParsingError
from knowledge_synthesizer.domain.models import FileSource, RawDocument

pytestmark = pytest.mark.unit

_MARKDOWN = b"""# Italy

Rome is the capital of Italy. It has a long and storied history spanning millennia.

## Cuisine

Italian cuisine is famous worldwide, with dishes like pasta and pizza.
"""

_HTML = b"""<!DOCTYPE html><html><head><title>Report</title></head>
<body><h1>Report</h1><p>Venice is a city built on water in northern Italy.</p></body></html>"""


class _CountingConverter:
    """Wraps a real DocumentConverter and counts how often it actually converts."""

    def __init__(self) -> None:
        self._inner = DocumentConverter()
        self.calls = 0

    def convert(self, source: DocumentStream) -> ConversionResult:
        self.calls += 1
        return self._inner.convert(source)


def _raw(content: bytes, mime: str, name: str) -> RawDocument:
    return RawDocument(content=content, mime_type=mime, source=FileSource(path=f"/{name}"))


def _parser(cache: DoclingDocumentCache, **kw: object) -> DoclingParser:
    return DoclingParser(cache=cache, **kw)  # type: ignore[arg-type]


def test_parse_markdown_produces_normalized_document() -> None:
    raw = _raw(_MARKDOWN, "text/markdown", "italy.md")
    parsed = _parser(DoclingDocumentCache()).parse(raw)

    assert "Rome is the capital of Italy" in parsed.markdown
    assert parsed.title == "Italy"
    assert parsed.content_hash == raw.content_hash
    assert parsed.source.uri == "/italy.md"
    assert parsed.page_count is None


def test_parse_html_extracts_text() -> None:
    raw = _raw(_HTML, "text/html", "report.html")
    parsed = _parser(DoclingDocumentCache()).parse(raw)
    assert "Venice" in parsed.markdown


def test_parse_generated_docx() -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Quarterly Report", level=1)
    document.add_paragraph("Milan hosted the annual conference this year.")
    buffer = io.BytesIO()
    document.save(buffer)

    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    parsed = _parser(DoclingDocumentCache()).parse(_raw(buffer.getvalue(), mime, "report.docx"))
    assert "Milan" in parsed.markdown


def test_unsupported_mime_raises_parsing_error() -> None:
    with pytest.raises(ParsingError):
        _parser(DoclingDocumentCache()).parse(_raw(b"x", "application/x-tar", "a.tar"))


def test_in_memory_cache_avoids_reparsing() -> None:
    converter = _CountingConverter()
    parser = _parser(DoclingDocumentCache(), converter=converter)
    raw = _raw(_MARKDOWN, "text/markdown", "italy.md")

    parser.parse(raw)
    parser.parse(raw)

    assert converter.calls == 1


def test_disk_cache_is_reused_by_a_fresh_parser(tmp_path: Path) -> None:
    raw = _raw(_MARKDOWN, "text/markdown", "italy.md")

    first = _CountingConverter()
    _parser(DoclingDocumentCache(cache_dir=tmp_path), converter=first).parse(raw)
    assert first.calls == 1
    assert (tmp_path / f"{raw.content_hash}.json").exists()

    second = _CountingConverter()
    _parser(DoclingDocumentCache(cache_dir=tmp_path), converter=second).parse(raw)
    assert second.calls == 0  # served from disk


def test_no_cache_does_not_write_to_disk(tmp_path: Path) -> None:
    raw = _raw(_MARKDOWN, "text/markdown", "italy.md")
    parser = _parser(DoclingDocumentCache(cache_dir=tmp_path), use_cache=False)
    parser.parse(raw)
    assert list(tmp_path.iterdir()) == []


def test_chunker_emits_chunks_with_provenance() -> None:
    cache = DoclingDocumentCache()
    raw = _raw(_MARKDOWN, "text/markdown", "italy.md")
    parsed = _parser(cache).parse(raw)

    chunker = DoclingChunker(cache=cache, chunker=HierarchicalChunker())
    chunks = chunker.chunk(parsed)

    assert chunks
    assert all(c.document_hash == raw.content_hash for c in chunks)
    assert all(c.provenance.source_uri == "/italy.md" for c in chunks)
    assert chunks[0].chunk_id != chunks[-1].chunk_id
    assert any("Rome is the capital" in c.text for c in chunks)
    assert any(c.provenance.section and "Italy" in c.provenance.section for c in chunks)


def test_chunker_without_cached_document_raises() -> None:
    raw = _raw(_MARKDOWN, "text/markdown", "italy.md")
    parsed = _parser(DoclingDocumentCache()).parse(raw)
    # A different cache has never seen this document.
    chunker = DoclingChunker(cache=DoclingDocumentCache(), chunker=HierarchicalChunker())
    with pytest.raises(ChunkingError):
        chunker.chunk(parsed)
