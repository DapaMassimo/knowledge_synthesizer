"""Load raw bytes from the local filesystem behind the SourceLoader port."""

from __future__ import annotations

import asyncio
import mimetypes
from pathlib import Path

from knowledge_synthesizer.domain.errors import SourceLoadError
from knowledge_synthesizer.domain.models import FileSource, RawDocument, Source

# Office formats mimetypes doesn't reliably know across platforms.
_EXTENSION_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class FileLoader:
    """Reads a local file into a single RawDocument. Implements SourceLoader."""

    async def load(self, source: Source) -> list[RawDocument]:
        if not isinstance(source, FileSource):
            raise SourceLoadError(f"FileLoader cannot load source of kind {source.kind!r}")
        path = Path(source.path)
        try:
            content = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise SourceLoadError(f"could not read file {source.path!r}: {exc}") from exc
        return [RawDocument(content=content, mime_type=self._mime_type(path), source=source)]

    @staticmethod
    def _mime_type(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in _EXTENSION_MIME:
            return _EXTENSION_MIME[suffix]
        guessed, _ = mimetypes.guess_type(path.name)
        return guessed or "application/octet-stream"
