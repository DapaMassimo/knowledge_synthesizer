import pytest

from knowledge_synthesizer.adapters.loaders.routing_loader import RoutingSourceLoader
from knowledge_synthesizer.domain.models import FileSource, RawDocument, Source, WebSource

pytestmark = pytest.mark.unit


class _RecordingLoader:
    def __init__(self, label: str) -> None:
        self.label = label
        self.seen: list[Source] = []

    async def load(self, source: Source) -> list[RawDocument]:
        self.seen.append(source)
        return [RawDocument(content=self.label.encode(), mime_type="text/plain", source=source)]


async def test_routes_each_source_kind_to_its_loader() -> None:
    file_loader = _RecordingLoader("file")
    web_loader = _RecordingLoader("web")
    router = RoutingSourceLoader(file_loader=file_loader, web_loader=web_loader)

    [from_file] = await router.load(FileSource(path="/a.txt"))
    [from_web] = await router.load(WebSource(url="https://example.com"))

    assert from_file.content == b"file"
    assert from_web.content == b"web"
    assert len(file_loader.seen) == 1
    assert len(web_loader.seen) == 1
