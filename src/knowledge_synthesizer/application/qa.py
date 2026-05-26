"""QA use case: retrieve -> build context -> generate -> attach citations.

Plain orchestration over the Retriever and LLMProvider ports (the graph-shaped control
flow a LangGraph would encode), kept framework-free so the application layer depends only
on domain abstractions.
"""

from __future__ import annotations

import asyncio
import re

from knowledge_synthesizer.domain.models import Answer, Citation, RetrievedChunk
from knowledge_synthesizer.domain.ports import LLMProvider, Retriever

_QA_SYSTEM = (
    "You answer questions strictly from the provided context. Cite the sources you use "
    "inline with their bracket numbers like [1]. If the context does not contain the "
    "answer, say you don't know. Do not invent facts or citations."
)
_NO_CONTEXT = "I don't have enough information in the provided sources to answer that."
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class QAService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        k: int = 5,
        snippet_chars: int = 200,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k
        self._snippet_chars = snippet_chars

    async def ask(self, question: str) -> Answer:
        # Retrieval + the LLM call are blocking; keep them off the event loop.
        return await asyncio.to_thread(self._answer, question)

    def _answer(self, question: str) -> Answer:
        retrieved = self._retriever.retrieve(question, self._k)
        if not retrieved:
            return Answer(question=question, text=_NO_CONTEXT, citations=[])
        completion = self._llm.complete(_QA_SYSTEM, self._build_prompt(question, retrieved)).strip()
        return Answer(
            question=question,
            text=completion,
            citations=self._citations(completion, retrieved),
        )

    def _build_prompt(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        blocks = [
            f"[{index}] {self._location(chunk)}\n{chunk.chunk.text}"
            for index, chunk in enumerate(retrieved, start=1)
        ]
        context = "\n\n".join(blocks)
        return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with inline [n] citations:"

    @staticmethod
    def _location(retrieved: RetrievedChunk) -> str:
        provenance = retrieved.chunk.provenance
        parts = [f"source: {provenance.source_uri}"]
        if provenance.page is not None:
            parts.append(f"page {provenance.page}")
        if provenance.section:
            parts.append(f"section: {provenance.section}")
        return "(" + ", ".join(parts) + ")"

    def _citations(self, completion: str, retrieved: list[RetrievedChunk]) -> list[Citation]:
        cited = self._cited_indices(completion, len(retrieved))
        chosen = cited if cited else list(range(len(retrieved)))
        return [
            Citation.from_chunk(retrieved[index].chunk, snippet=self._snippet(retrieved[index]))
            for index in chosen
        ]

    @staticmethod
    def _cited_indices(completion: str, count: int) -> list[int]:
        indices: list[int] = []
        for match in _CITATION_PATTERN.findall(completion):
            index = int(match) - 1
            if 0 <= index < count and index not in indices:
                indices.append(index)
        return indices

    def _snippet(self, retrieved: RetrievedChunk) -> str:
        text = retrieved.chunk.text.strip()
        if len(text) <= self._snippet_chars:
            return text
        return text[: self._snippet_chars] + "…"
