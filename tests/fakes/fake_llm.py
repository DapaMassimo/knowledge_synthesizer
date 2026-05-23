"""Deterministic LLM fake for unit tests."""

from __future__ import annotations

from collections.abc import Callable


class FakeLLM:
    """Implements the LLMProvider port. Returns a canned response, a responder, or echoes."""

    def __init__(
        self,
        response: str | None = None,
        responder: Callable[[str, str], str] | None = None,
    ) -> None:
        self._response = response
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if self._responder is not None:
            return self._responder(system, prompt)
        if self._response is not None:
            return self._response
        return prompt
