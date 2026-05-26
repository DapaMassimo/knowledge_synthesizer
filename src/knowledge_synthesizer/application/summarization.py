"""Summarization use case: map-reduce over a topic's chunks, behind a Summarizer strategy.

`MapReduceSummarizer` (the default strategy) summarizes each chunk (map) then folds the
partial summaries into one structured Summary (reduce). `SummarizationService` gathers the
topic's chunks via the Retriever and delegates to the injected Summarizer, so a hierarchical
strategy can replace map-reduce without touching callers.
"""

from __future__ import annotations

import asyncio

from knowledge_synthesizer.domain.models import Chunk, Citation, Summary
from knowledge_synthesizer.domain.ports import LLMProvider, Retriever, Summarizer

_MAP_SYSTEM = (
    "You extract what a single passage says about a topic. Reply with one or two factual "
    "sentences, or exactly 'NONE' if the passage is irrelevant to the topic."
)
_REDUCE_SYSTEM = (
    "You synthesize notes into a structured topic summary. Respond exactly as:\n"
    "Overview: <one short paragraph>\nKey points:\n- <point>\n- <point>"
)
_NONE = "NONE"


class MapReduceSummarizer:
    """Default Summarizer strategy: map each chunk to a partial, then reduce to a Summary."""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def summarize(self, topic: str, chunks: list[Chunk]) -> Summary:
        contributions: list[tuple[Chunk, str]] = []
        for chunk in chunks:
            partial = self._map(topic, chunk)
            if partial and partial.upper() != _NONE:
                contributions.append((chunk, partial))
        if not contributions:
            return Summary(
                topic=topic,
                overview="The indexed sources do not cover this topic.",
                key_points=[],
                citations=[],
            )
        return self._reduce(topic, contributions)

    def _map(self, topic: str, chunk: Chunk) -> str:
        prompt = (
            f"Topic: {topic}\n\nPassage:\n{chunk.text}\n\nSummary (one or two sentences, or NONE):"
        )
        return self._llm.complete(_MAP_SYSTEM, prompt).strip()

    def _reduce(self, topic: str, contributions: list[tuple[Chunk, str]]) -> Summary:
        notes = "\n".join(f"- {partial}" for _, partial in contributions)
        prompt = f"Topic: {topic}\n\nNotes:\n{notes}\n\nSummary:"
        completion = self._llm.complete(_REDUCE_SYSTEM, prompt)
        overview, key_points = _parse_summary(completion)
        return Summary(
            topic=topic,
            overview=overview,
            key_points=key_points,
            citations=[Citation.from_chunk(chunk) for chunk, _ in contributions],
        )


class SummarizationService:
    def __init__(self, retriever: Retriever, summarizer: Summarizer, k: int = 12) -> None:
        self._retriever = retriever
        self._summarizer = summarizer
        self._k = k

    async def summarize(self, topic: str) -> Summary:
        return await asyncio.to_thread(self._summarize, topic)

    def _summarize(self, topic: str) -> Summary:
        chunks = [hit.chunk for hit in self._retriever.retrieve(topic, self._k)]
        if not chunks:
            return Summary(
                topic=topic,
                overview="No sources are indexed for this topic.",
                key_points=[],
                citations=[],
            )
        return self._summarizer.summarize(topic, chunks)


def _parse_summary(text: str) -> tuple[str, list[str]]:
    overview_lines: list[str] = []
    key_points: list[str] = []
    in_points = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("overview:"):
            overview_lines.append(line[len("overview:") :].strip())
            in_points = False
        elif lower.startswith("key point"):
            in_points = True
        elif line[0] in "-*•":
            point = line.lstrip("-*• \t").strip()
            if point:
                key_points.append(point)
        elif not in_points:
            overview_lines.append(line)
    overview = " ".join(part for part in overview_lines if part).strip() or text.strip()
    return overview, key_points
