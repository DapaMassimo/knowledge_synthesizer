"""OpenAI embeddings adapter (text-embedding-3-small). Implements the EmbeddingModel port.

Vectors are computed here and passed to the vector store already prepared; the store's
built-in embedding function is never used.
"""

from __future__ import annotations

import logging

from openai import OpenAI, OpenAIError

from knowledge_synthesizer.domain.errors import EmbeddingError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbeddings:
    def __init__(
        self,
        client: OpenAI,
        model: str = _DEFAULT_MODEL,
        batch_size: int = 128,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        if texts:
            logger.debug(
                "Embedding %d text(s) with %s in batches of %d",
                len(texts),
                self._model,
                self._batch_size,
            )
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.embeddings.create(model=self._model, input=batch)
            except OpenAIError as exc:
                logger.warning("Embedding request failed: %s", exc)
                raise EmbeddingError(f"OpenAI embeddings request failed: {exc}") from exc
            for item in sorted(response.data, key=lambda datum: datum.index):
                vectors.append(list(item.embedding))
        return vectors
