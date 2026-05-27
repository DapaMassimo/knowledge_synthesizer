"""OpenAI chat LLM adapter (gpt-4o-mini). Implements the LLMProvider port."""

from __future__ import annotations

from typing import Any, cast

from openai import OpenAI, OpenAIError

from knowledge_synthesizer.domain.errors import LLMError

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
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=cast("Any", messages),
            )
        except OpenAIError as exc:
            raise LLMError(f"OpenAI completion failed: {exc}") from exc
        return response.choices[0].message.content or ""
