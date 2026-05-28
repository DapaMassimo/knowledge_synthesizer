from pathlib import Path

import pytest

from knowledge_synthesizer.domain.models import (
    Answer,
    Citation,
    FileSource,
    Summary,
    WebSource,
)
from knowledge_synthesizer.entrypoints.presentation import (
    answer_markdown,
    citation_markdown,
    citations_markdown,
    materialize_uploads,
    parse_source,
    parse_sources,
    summary_markdown,
)

pytestmark = pytest.mark.unit


def test_parse_source_detects_kind_and_strips() -> None:
    assert isinstance(parse_source("https://example.com"), WebSource)
    assert isinstance(parse_source("  /a.pdf  "), FileSource)


def test_parse_sources_skips_blank_lines() -> None:
    sources = parse_sources("/a.pdf\n\n   \nhttps://example.com\n")
    assert len(sources) == 2
    assert isinstance(sources[0], FileSource)
    assert isinstance(sources[1], WebSource)


def test_materialize_uploads_writes_files_and_returns_sources(tmp_path: Path) -> None:
    dest = tmp_path / "uploads"
    sources = materialize_uploads([("a.pdf", b"PDF"), ("notes.md", b"# Hi")], dest)

    assert all(isinstance(source, FileSource) for source in sources)
    assert (dest / "a.pdf").read_bytes() == b"PDF"
    assert (dest / "notes.md").read_bytes() == b"# Hi"
    assert {Path(s.path).name for s in sources} == {"a.pdf", "notes.md"}  # type: ignore[attr-defined]


def test_materialize_uploads_strips_directory_components(tmp_path: Path) -> None:
    dest = tmp_path / "uploads"
    [source] = materialize_uploads([("../../etc/evil.txt", b"x")], dest)

    written = Path(source.path)  # type: ignore[attr-defined]
    assert written.parent == dest  # cannot escape dest_dir
    assert written.name == "evil.txt"


def test_citation_markdown_links_web_but_not_files() -> None:
    web = citation_markdown(1, Citation(source_uri="https://example.com/a", page=2))
    assert "](https://example.com/a)" in web
    assert "p.2" in web

    file = citation_markdown(2, Citation(source_uri="/a.pdf", section="Intro"))
    assert file.startswith("[2] /a.pdf")
    assert "Intro" in file
    assert "](" not in file


def test_citations_markdown_empty_is_blank() -> None:
    assert citations_markdown([]) == ""


def test_answer_markdown_with_and_without_citations() -> None:
    with_cite = answer_markdown(
        Answer(question="q", text="Rome [1].", citations=[Citation(source_uri="/a.pdf")])
    )
    assert "Rome [1]." in with_cite
    assert "**Sources**" in with_cite
    assert "/a.pdf" in with_cite

    assert answer_markdown(Answer(question="q", text="No idea.", citations=[])) == "No idea."


def test_summary_markdown_renders_sections() -> None:
    markdown = summary_markdown(
        Summary(
            topic="Italy",
            overview="Italy is in southern Europe.",
            key_points=["Rome is the capital"],
            citations=[Citation(source_uri="https://example.com")],
        )
    )
    assert "## Italy" in markdown
    assert "Italy is in southern Europe." in markdown
    assert "- Rome is the capital" in markdown
    assert "**Sources**" in markdown
