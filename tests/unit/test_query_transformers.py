import pytest

from knowledge_synthesizer.adapters.retrieval.query_transformers import (
    HydeTransformer,
    MultiQueryTransformer,
    PassthroughTransformer,
)
from tests.fakes import FakeLLM

pytestmark = pytest.mark.unit


def test_passthrough_returns_question_unchanged() -> None:
    assert PassthroughTransformer().transform("what is rome") == ["what is rome"]


def test_multiquery_prepends_original_and_parses_lines() -> None:
    llm = FakeLLM(response="capital of italy\nrome history\nitalian geography")
    queries = MultiQueryTransformer(llm).transform("tell me about rome")
    assert queries[0] == "tell me about rome"
    assert "capital of italy" in queries
    assert "rome history" in queries


def test_multiquery_strips_list_markers_and_dedupes() -> None:
    llm = FakeLLM(response="1. rome\n- rome\n* venice")
    queries = MultiQueryTransformer(llm, n=5).transform("italy")
    assert queries == ["italy", "rome", "venice"]


def test_multiquery_limits_to_n_plus_original() -> None:
    llm = FakeLLM(response="a\nb\nc\nd\ne")
    queries = MultiQueryTransformer(llm, n=2).transform("q")
    assert len(queries) == 3  # original + 2


def test_hyde_returns_hypothetical_passage() -> None:
    llm = FakeLLM(response="Rome is the capital of Italy and was founded in antiquity.")
    assert HydeTransformer(llm).transform("capital of italy") == [
        "Rome is the capital of Italy and was founded in antiquity."
    ]


def test_hyde_falls_back_to_question_when_empty() -> None:
    assert HydeTransformer(FakeLLM(response="   ")).transform("capital of italy") == [
        "capital of italy"
    ]
