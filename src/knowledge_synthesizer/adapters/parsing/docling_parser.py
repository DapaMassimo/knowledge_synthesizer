"""Parse raw bytes into a normalized ParsedDocument using Docling. Implements DocumentParser."""

from __future__ import annotations

import io

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument  # type: ignore[attr-defined]
from docling_core.types.io import DocumentStream

from knowledge_synthesizer.adapters.parsing.docling_cache import DoclingDocumentCache
from knowledge_synthesizer.domain.errors import ParsingError
from knowledge_synthesizer.domain.models import ParsedDocument, RawDocument

_MIME_TO_SUFFIX: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "text/markdown": ".md",
    "text/plain": ".md",
    "text/html": ".html",
}


def _title_from_markdown(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip() or None
    return None


class DoclingParser:
    """Converts a RawDocument to a ParsedDocument and caches the native DoclingDocument.

    The DoclingDocument is stored in the shared cache keyed by ``content_hash`` so the
    chunker can run Docling's chunker over the same structure without re-parsing.
    """

    def __init__(
        self,
        cache: DoclingDocumentCache,
        converter: DocumentConverter | None = None,
        use_cache: bool = True,
    ) -> None:
        self._cache = cache
        self._converter = converter if converter is not None else DocumentConverter()
        self._use_cache = use_cache

    def parse(self, raw: RawDocument) -> ParsedDocument:
        document = self._cache.get(raw.content_hash, use_disk=self._use_cache)
        if document is None:
            document = self._convert(raw)
        # Always store in memory (parser -> chunker hand-off); disk gated by use_cache.
        self._cache.put(raw.content_hash, document, use_disk=self._use_cache)

        markdown = document.export_to_markdown()
        return ParsedDocument(
            source=raw.source,
            content_hash=raw.content_hash,
            markdown=markdown,
            title=_title_from_markdown(markdown),
            page_count=document.num_pages() or None,  # type: ignore[no-untyped-call]
        )

    def _convert(self, raw: RawDocument) -> DoclingDocument:
        suffix = _MIME_TO_SUFFIX.get(raw.mime_type)
        if suffix is None:
            raise ParsingError(f"unsupported mime type for parsing: {raw.mime_type!r}")
        stream = DocumentStream(name=f"document{suffix}", stream=io.BytesIO(raw.content))
        try:
            result = self._converter.convert(stream)
        except Exception as exc:
            raise ParsingError(f"docling failed to parse {raw.source.uri!r}: {exc}") from exc
        return result.document
