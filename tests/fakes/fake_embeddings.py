"""Deterministic embedding fake: L2-normalized bag-of-tokens vectors."""

from __future__ import annotations

import hashlib
import math


class FakeEmbeddings:
    """Implements the EmbeddingModel port. Same text -> same vector; cosine is meaningful."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            bucket = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0.0:
            return vec
        return [value / norm for value in vec]
