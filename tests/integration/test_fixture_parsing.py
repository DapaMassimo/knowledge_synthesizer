"""Offline golden tests over real sample documents in tests/fixtures/.

The PDF needs Docling's layout/table models (download once via
`uv run python -c "from docling.utils.model_downloader import download_models; download_models()"`);
the test loads them from the local cache via ``artifacts_path`` and forces HF/transformers
offline to prove no network is touched. PPTX needs no models. Both skip if the fixture (or,
for the PDF, the models) is missing.
"""

from pathlib import Path

import pytest
from docling.chunking import HierarchicalChunker
from docling.datamodel.settings import settings as docling_settings

from knowledge_synthesizer.adapters.parsing.docling_cache import DoclingDocumentCache
from knowledge_synthesizer.adapters.parsing.docling_chunker import DoclingChunker
from knowledge_synthesizer.adapters.parsing.docling_parser import DoclingParser
from knowledge_synthesizer.domain.models import Chunk, FileSource, ParsedDocument, RawDocument

pytestmark = pytest.mark.integration

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_MODELS = docling_settings.cache_dir / "models"
_PDF = _FIXTURES / "sample.pdf"
_PPTX = _FIXTURES / "sample.pptx"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def _raw(path: Path, mime: str) -> RawDocument:
    return RawDocument(content=path.read_bytes(), mime_type=mime, source=FileSource(path=str(path)))


def _parse_and_chunk(
    parser: DoclingParser, cache: DoclingDocumentCache, raw: RawDocument
) -> tuple[ParsedDocument, list[Chunk]]:
    parsed = parser.parse(raw)
    chunks = DoclingChunker(cache=cache, chunker=HierarchicalChunker()).chunk(parsed)
    return parsed, chunks


@pytest.mark.skipif(not _PDF.exists(), reason="tests/fixtures/sample.pdf missing")
@pytest.mark.skipif(not _MODELS.exists(), reason="Docling models not downloaded")
def test_pdf_fixture_parses_offline_with_page_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Prove no network: models must come from the local cache.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    cache = DoclingDocumentCache()
    parser = DoclingParser(cache=cache, artifacts_path=_MODELS, do_ocr=False)
    parsed, chunks = _parse_and_chunk(parser, cache, _raw(_PDF, "application/pdf"))

    assert parsed.page_count == 10
    assert "SAS" in parsed.markdown
    assert "Azure Kubernetes Service" in parsed.markdown
    assert chunks
    assert all(chunk.document_hash == parsed.content_hash for chunk in chunks)

    pages = {chunk.provenance.page for chunk in chunks if chunk.provenance.page is not None}
    assert pages, "PDF chunks should carry page provenance"
    assert min(pages) >= 1
    assert max(pages) <= 10


@pytest.mark.skipif(not _PPTX.exists(), reason="tests/fixtures/sample.pptx missing")
def test_pptx_fixture_parses_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # PPTX needs no models; assert offline anyway

    cache = DoclingDocumentCache()
    parsed, chunks = _parse_and_chunk(
        parser=DoclingParser(cache=cache), cache=cache, raw=_raw(_PPTX, _PPTX_MIME)
    )

    assert "LOREM IPSUM" in parsed.markdown.upper()
    assert chunks
    assert all(chunk.provenance.source_uri.endswith("sample.pptx") for chunk in chunks)
