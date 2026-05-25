"""Query transformers implementing the QueryTransformer port.

- Passthrough: the question unchanged (similarity-only baseline / ablation).
- MultiQuery (default): the LLM rewrites the question into several diverse search queries.
- HyDE: the LLM writes a hypothetical answer passage; that passage is what gets embedded.
"""

from __future__ import annotations

from knowledge_synthesizer.domain.ports import LLMProvider

_MULTIQUERY_SYSTEM = (
    "You rewrite a user question into diverse, standalone search queries that surface "
    "relevant passages. Output one query per line and nothing else."
)
_HYDE_SYSTEM = (
    "You write a short, factual passage that could plausibly answer the user's question. "
    "Write only the passage."
)
_LIST_NOISE = "-*0123456789.) \t"


class PassthroughTransformer:
    def transform(self, question: str) -> list[str]:
        return [question]


class MultiQueryTransformer:
    def __init__(self, llm: LLMProvider, n: int = 3) -> None:
        self._llm = llm
        self._n = n

    def transform(self, question: str) -> list[str]:
        prompt = (
            f"Generate {self._n} different search queries for the question below.\n\n"
            f"Question: {question}\nQueries:"
        )
        completion = self._llm.complete(_MULTIQUERY_SYSTEM, prompt)
        queries = [question]
        for line in completion.splitlines():
            cleaned = line.strip().lstrip(_LIST_NOISE).strip()
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
        return queries[: self._n + 1]


class HydeTransformer:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def transform(self, question: str) -> list[str]:
        prompt = f"Question: {question}\nPassage:"
        passage = self._llm.complete(_HYDE_SYSTEM, prompt).strip()
        return [passage] if passage else [question]
