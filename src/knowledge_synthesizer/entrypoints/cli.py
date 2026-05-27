"""CLI entrypoint: index / ask / summarize. Calls the container, then the use cases."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Sequence

from knowledge_synthesizer.application.indexing import IndexReport
from knowledge_synthesizer.composition.container import Container
from knowledge_synthesizer.config.settings import Settings
from knowledge_synthesizer.domain.models import Answer, Citation, Summary
from knowledge_synthesizer.entrypoints.presentation import parse_source

Emit = Callable[[str], None]


def _citation_lines(citations: list[Citation]) -> list[str]:
    lines = []
    for index, citation in enumerate(citations, start=1):
        location = citation.source_uri
        if citation.page is not None:
            location += f", p.{citation.page}"
        if citation.section:
            location += f", {citation.section}"
        lines.append(f"  [{index}] {location}")
    return lines


def _emit_report(report: IndexReport, emit: Emit) -> None:
    emit(
        f"Indexed {report.documents_indexed} document(s), "
        f"skipped {report.documents_skipped}, {report.chunks_indexed} chunk(s)."
    )


def _emit_answer(answer: Answer, emit: Emit) -> None:
    emit(answer.text)
    if answer.citations:
        emit("\nSources:")
        for line in _citation_lines(answer.citations):
            emit(line)


def _emit_summary(summary: Summary, emit: Emit) -> None:
    emit(f"# {summary.topic}\n")
    emit(summary.overview)
    if summary.key_points:
        emit("\nKey points:")
        for point in summary.key_points:
            emit(f"  - {point}")
    if summary.citations:
        emit("\nSources:")
        for line in _citation_lines(summary.citations):
            emit(line)


def _dispatch(args: argparse.Namespace, container: Container, emit: Emit) -> int:
    if args.command == "index":
        sources = [parse_source(value) for value in args.sources]
        _emit_report(asyncio.run(container.indexing_service().index(sources)), emit)
    elif args.command == "ask":
        _emit_answer(asyncio.run(container.qa_service().ask(args.question)), emit)
    elif args.command == "summarize":
        _emit_summary(asyncio.run(container.summarization_service().summarize(args.topic)), emit)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge-synthesizer",
        description="Index sources and run grounded QA and summarization over them.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index sources (file paths or URLs).")
    index_parser.add_argument("sources", nargs="+", help="File paths or http(s) URLs.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question grounded in the sources.")
    ask_parser.add_argument("question", help="The question to answer.")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize a topic.")
    summarize_parser.add_argument("topic", help="The topic to summarize.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    container = Container(Settings())
    return _dispatch(args, container, print)


if __name__ == "__main__":
    raise SystemExit(main())
