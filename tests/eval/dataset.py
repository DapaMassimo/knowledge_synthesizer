"""A small fixed evaluation dataset: a tiny corpus plus QA samples with ground truths."""

from __future__ import annotations

from dataclasses import dataclass

EVAL_CORPUS: list[tuple[str, str]] = [
    (
        "/italy.md",
        "# Italy\n\nRome is the capital of Italy. Italy is a country in southern Europe.\n\n"
        "## Cuisine\n\nItalian cuisine is known worldwide for pasta and pizza.",
    ),
    (
        "/france.md",
        "# France\n\nParis is the capital of France. France is a country in western Europe.",
    ),
]


@dataclass(frozen=True)
class EvalSample:
    question: str
    ground_truth: str


EVAL_SAMPLES: list[EvalSample] = [
    EvalSample("What is the capital of Italy?", "Rome is the capital of Italy."),
    EvalSample("What is the capital of France?", "Paris is the capital of France."),
    EvalSample(
        "What is Italian cuisine known for?", "Italian cuisine is known for pasta and pizza."
    ),
]
