"""Presentation helpers shared by entrypoints (no UI framework imports)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from knowledge_synthesizer.domain.models import (
    Answer,
    Citation,
    FileSource,
    Source,
    Summary,
    WebSource,
)

_URL_PREFIXES = ("http://", "https://")
_SECRET_HINTS = ("key", "token", "secret", "password")

KNOWN_LLM_MODELS = ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1")
KNOWN_EMBEDDING_MODELS = ("text-embedding-3-small", "text-embedding-3-large")


def model_options(default: str, known: Iterable[str]) -> list[str]:
    """Return selectable models with the configured default first (deduplicated)."""
    options = [default] if default else []
    for model in known:
        if model not in options:
            options.append(model)
    return options


def conversation_markdown(answers: list[Answer]) -> str:
    """Render a whole Q&A conversation as a downloadable markdown transcript."""
    blocks = ["# Conversation"]
    for answer in answers:
        blocks.append(f"## {answer.question}")
        blocks.append(answer_markdown(answer))
    return "\n\n".join(blocks)


def parse_source(value: str) -> Source:
    value = value.strip()
    if value.startswith(_URL_PREFIXES):
        return WebSource(url=value)
    return FileSource(path=value)


def parse_sources(raw: str) -> list[Source]:
    return [parse_source(line) for line in raw.splitlines() if line.strip()]


def mask_secret(value: str) -> str:
    """Show enough of a secret to identify it, without exposing it (prefix…suffix + length)."""
    value = value.strip()
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return f"…{value[-2:]} (len {len(value)})"
    return f"{value[:7]}…{value[-4:]} (len {len(value)})"


def settings_rows(values: Mapping[str, object]) -> list[tuple[str, str]]:
    """Render config settings as (name, value) rows, masking secret-looking fields."""
    rows: list[tuple[str, str]] = []
    for name in sorted(values):
        raw = values[name]
        if isinstance(raw, str) and any(hint in name.lower() for hint in _SECRET_HINTS):
            rows.append((name, mask_secret(raw)))
        else:
            rows.append((name, str(raw)))
    return rows


def materialize_uploads(items: list[tuple[str, bytes]], dest_dir: Path) -> list[Source]:
    """Persist uploaded (filename, bytes) pairs under dest_dir and return file sources.

    The filename is reduced to its basename so an upload can't write outside dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    sources: list[Source] = []
    for name, data in items:
        safe_name = Path(name).name or "upload"
        path = dest_dir / safe_name
        path.write_bytes(data)
        sources.append(FileSource(path=str(path)))
    return sources


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
