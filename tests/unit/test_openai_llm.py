from types import SimpleNamespace
from typing import Any

import pytest
from openai import OpenAIError

from knowledge_synthesizer.adapters.llm.openai_llm import OpenAILLM
from knowledge_synthesizer.domain.errors import LLMError

pytestmark = pytest.mark.unit


class _FakeCompletions:
    def __init__(self, content: str | None, *, fail: bool = False) -> None:
        self._content = content
        self._fail = fail
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        if self._fail:
            raise OpenAIError("boom")
        self.kwargs = kwargs
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, content: str | None = "hello", *, fail: bool = False) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content, fail=fail))


def test_complete_returns_message_content() -> None:
    client = _FakeClient(content="Rome is the capital.")
    result = OpenAILLM(client, model="gpt-4o-mini").complete("sys", "prompt")  # type: ignore[arg-type]
    assert result == "Rome is the capital."


def test_complete_sends_system_and_user_messages() -> None:
    client = _FakeClient()
    OpenAILLM(client).complete("the system", "the prompt")  # type: ignore[arg-type]
    messages = client.chat.completions.kwargs["messages"]
    assert messages == [
        {"role": "system", "content": "the system"},
        {"role": "user", "content": "the prompt"},
    ]


def test_complete_handles_none_content() -> None:
    client = _FakeClient(content=None)
    assert OpenAILLM(client).complete("s", "p") == ""  # type: ignore[arg-type]


def test_complete_wraps_api_errors() -> None:
    client = _FakeClient(fail=True)
    with pytest.raises(LLMError):
        OpenAILLM(client).complete("s", "p")  # type: ignore[arg-type]
