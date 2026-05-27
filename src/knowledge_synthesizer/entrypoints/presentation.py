"""Pure presentation helpers shared by entrypoints (no UI framework imports)."""

from __future__ import annotations

from knowledge_synthesizer.domain.models import (
    Answer,
    Citation,
    FileSource,
    Source,
    Summary,
    WebSource,
)

_URL_PREFIXES = ("http://", "https://")


def parse_source(value: str) -> Source:
    value = value.strip()
    if value.startswith(_URL_PREFIXES):
        return WebSource(url=value)
    return FileSource(path=value)


def parse_sources(raw: str) -> list[Source]:
    return [parse_source(line) for line in raw.splitlines() if line.strip()]


def citation_markdown(index: int, citation: Citation) -> str:
    label = citation.source_uri
    if citation.page is not None:
        label += f", p.{citation.page}"
    if citation.section:
        label += f" — {citation.section}"
    if citation.source_uri.startswith(_URL_PREFIXES):
        return f"[{index}] [{label}]({citation.source_uri})"
    return f"[{index}] {label}"


def citations_markdown(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = "\n".join(
        citation_markdown(index, citation) for index, citation in enumerate(citations, start=1)
    )
    return f"**Sources**\n\n{lines}"


def answer_markdown(answer: Answer) -> str:
    blocks = [answer.text]
    citations = citations_markdown(answer.citations)
    if citations:
        blocks.append(citations)
    return "\n\n".join(blocks)


def summary_markdown(summary: Summary) -> str:
    blocks = [f"## {summary.topic}", summary.overview]
    if summary.key_points:
        blocks.append("\n".join(f"- {point}" for point in summary.key_points))
    citations = citations_markdown(summary.citations)
    if citations:
        blocks.append(citations)
    return "\n\n".join(blocks)
