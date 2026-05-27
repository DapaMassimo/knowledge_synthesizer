import pytest

from knowledge_synthesizer.application.indexing import IndexReport
from knowledge_synthesizer.domain.models import Answer, Citation, FileSource, Summary, WebSource
from knowledge_synthesizer.entrypoints.cli import _dispatch, build_parser

pytestmark = pytest.mark.unit


class _StubService:
    def __init__(self, result: object) -> None:
        self.result = result
        self.arg: object = None

    async def index(self, sources: object) -> object:
        self.arg = sources
        return self.result

    async def ask(self, question: object) -> object:
        self.arg = question
        return self.result

    async def summarize(self, topic: object) -> object:
        self.arg = topic
        return self.result


class _StubContainer:
    def __init__(
        self,
        indexing: _StubService | None = None,
        qa: _StubService | None = None,
        summarization: _StubService | None = None,
    ) -> None:
        self._indexing = indexing
        self._qa = qa
        self._summarization = summarization

    def indexing_service(self) -> _StubService | None:
        return self._indexing

    def qa_service(self) -> _StubService | None:
        return self._qa

    def summarization_service(self) -> _StubService | None:
        return self._summarization


def test_dispatch_index_parses_sources_and_reports() -> None:
    service = _StubService(IndexReport(documents_indexed=2, documents_skipped=1, chunks_indexed=5))
    container = _StubContainer(indexing=service)
    args = build_parser().parse_args(["index", "/a.txt", "https://example.com"])
    lines: list[str] = []

    _dispatch(args, container, lines.append)  # type: ignore[arg-type]

    assert any("Indexed 2 document(s), skipped 1, 5 chunk(s)." in line for line in lines)
    assert isinstance(service.arg[0], FileSource)  # type: ignore[index]
    assert isinstance(service.arg[1], WebSource)  # type: ignore[index]


def test_dispatch_ask_prints_answer_and_sources() -> None:
    answer = Answer(
        question="q",
        text="Rome is the capital [1].",
        citations=[Citation(source_uri="/italy.pdf", page=3)],
    )
    container = _StubContainer(qa=_StubService(answer))
    args = build_parser().parse_args(["ask", "capital of italy?"])
    lines: list[str] = []

    _dispatch(args, container, lines.append)  # type: ignore[arg-type]

    assert "Rome is the capital [1]." in lines
    assert any("/italy.pdf, p.3" in line for line in lines)


def test_dispatch_summarize_prints_overview_points_sources() -> None:
    summary = Summary(
        topic="Italy",
        overview="Italy is in southern Europe.",
        key_points=["Rome is the capital"],
        citations=[Citation(source_uri="/italy.pdf")],
    )
    container = _StubContainer(summarization=_StubService(summary))
    args = build_parser().parse_args(["summarize", "Italy"])
    lines: list[str] = []

    _dispatch(args, container, lines.append)  # type: ignore[arg-type]

    assert any("# Italy" in line for line in lines)
    assert any("Rome is the capital" in line for line in lines)
    assert any("/italy.pdf" in line for line in lines)


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
