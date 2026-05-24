"""Dispatch a source to the right loader by kind. Implements SourceLoader."""

from __future__ import annotations

from knowledge_synthesizer.domain.errors import SourceLoadError
from knowledge_synthesizer.domain.models import FileSource, RawDocument, Source, WebSource
from knowledge_synthesizer.domain.ports import SourceLoader


class RoutingSourceLoader:
    """Routes file sources to a file loader and web sources to a web loader."""

    def __init__(self, file_loader: SourceLoader, web_loader: SourceLoader) -> None:
        self._file_loader = file_loader
        self._web_loader = web_loader

    async def load(self, source: Source) -> list[RawDocument]:
        if isinstance(source, FileSource):
            return await self._file_loader.load(source)
        if isinstance(source, WebSource):
            return await self._web_loader.load(source)
        raise SourceLoadError(f"no loader registered for source kind {source.kind!r}")
