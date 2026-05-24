"""Fetch a web page and extract its main content behind the SourceLoader port."""

from __future__ import annotations

import httpx
import trafilatura

from knowledge_synthesizer.domain.errors import SourceLoadError
from knowledge_synthesizer.domain.models import RawDocument, Source, WebSource

_WEB_MIME = "text/markdown"


class WebLoader:
    """Scrapes a URL with httpx and extracts the main text as markdown via trafilatura.

    An httpx.AsyncClient may be injected (constructor injection) so tests can supply a
    mock transport; otherwise a client is created per request. Implements SourceLoader.
    """

    def __init__(self, client: httpx.AsyncClient | None = None, timeout: float = 20.0) -> None:
        self._client = client
        self._timeout = timeout

    async def load(self, source: Source) -> list[RawDocument]:
        if not isinstance(source, WebSource):
            raise SourceLoadError(f"WebLoader cannot load source of kind {source.kind!r}")
        html = await self._fetch(source.url)
        markdown = self._extract(html, source.url)
        return [
            RawDocument(
                content=markdown.encode("utf-8"),
                mime_type=_WEB_MIME,
                source=source,
            )
        ]

    async def _fetch(self, url: str) -> str:
        try:
            if self._client is not None:
                response = await self._client.get(url, follow_redirects=True)
            else:
                async with httpx.AsyncClient(
                    follow_redirects=True, timeout=self._timeout
                ) as client:
                    response = await client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SourceLoadError(f"could not fetch {url!r}: {exc}") from exc
        return response.text

    @staticmethod
    def _extract(html: str, url: str) -> str:
        extracted = trafilatura.extract(
            html,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
        )
        if not extracted:
            raise SourceLoadError(f"no main content could be extracted from {url!r}")
        return str(extracted)
