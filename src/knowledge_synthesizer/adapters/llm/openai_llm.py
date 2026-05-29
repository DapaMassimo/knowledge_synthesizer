"""OpenAI chat LLM adapter (gpt-4o-mini). Implements the LLMProvider port."""

from __future__ import annotations

import logging
from typing import Any, cast

from openai import OpenAI, OpenAIError

from knowledge_synthesizer.domain.errors import LLMError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLM:
    def __init__(
        self,
        client: OpenAI,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        logger.info(
            "LLM call: model=%s, system=%d chars, prompt=%d chars",
            self._model,
            len(system),
            len(prompt),
        )
        logger.debug("LLM prompt:\n%s", prompt)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=cast("Any", messages),
            )
        except OpenAIError as exc:
            logger.warning("LLM call failed: %s", exc)
            raise LLMError(f"OpenAI completion failed: {exc}") from exc
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", None)
        logger.info(
            "LLM response: %d chars%s", len(content), f", {tokens} tokens" if tokens else ""
        )
        logger.debug("LLM response body:\n%s", content)
        return content
