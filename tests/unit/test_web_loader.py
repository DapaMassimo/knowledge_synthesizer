from collections.abc import Callable

import httpx
import pytest

from knowledge_synthesizer.adapters.loaders.web_loader import WebLoader
from knowledge_synthesizer.domain.errors import SourceLoadError
from knowledge_synthesizer.domain.models import FileSource, WebSource

pytestmark = pytest.mark.unit

_ARTICLE_HTML = """<!DOCTYPE html>
<html><head><title>Rome</title></head>
<body>
<nav>Home About Contact</nav>
<article>
<h1>Rome</h1>
<p>Rome is the capital of Italy and one of the oldest continuously inhabited cities in
Europe, with a history that spans more than two and a half thousand years.</p>
<p>The city is renowned for landmarks such as the Colosseum, the Roman Forum, and the
Pantheon, and it surrounds the independent state of Vatican City.</p>
<p>Today Rome is a major cultural, political, and tourist centre, drawing millions of
visitors who come to see its ancient monuments and Renaissance art.</p>
</article>
<footer>Copyright notice that should be stripped out.</footer>
</body></html>"""

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_extracts_main_content_as_markdown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=_ARTICLE_HTML)

    async with _client(handler) as client:
        [raw] = await WebLoader(client=client).load(WebSource(url="https://example.com/rome"))

    text = raw.content.decode("utf-8")
    assert raw.mime_type == "text/markdown"
    assert "Rome is the capital of Italy" in text
    assert "Copyright notice" not in text
    assert raw.source.uri == "https://example.com/rome"


async def test_http_error_raises_source_load_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with _client(handler) as client:
        with pytest.raises(SourceLoadError):
            await WebLoader(client=client).load(WebSource(url="https://example.com/missing"))


async def test_empty_page_raises_source_load_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<html><body></body></html>")

    async with _client(handler) as client:
        with pytest.raises(SourceLoadError):
            await WebLoader(client=client).load(WebSource(url="https://example.com/empty"))


async def test_rejects_file_source() -> None:
    with pytest.raises(SourceLoadError):
        await WebLoader().load(FileSource(path="/tmp/x.txt"))
