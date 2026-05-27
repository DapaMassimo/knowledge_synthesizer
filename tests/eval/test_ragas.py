"""Opt-in ragas evaluation (faithfulness, answer relevance, context precision/recall).

Runs only via `make test-eval` and only when ragas (and a working langchain stack) plus an
OPENAI_API_KEY are available. ragas pulls a langchain-openai stack that conflicts with the
project's openai>=2 SDK in some environments; the test skips cleanly in that case.
"""

from __future__ import annotations

import os

import pytest

from tests.eval.dataset import EVAL_CORPUS, EVAL_SAMPLES
from tests.eval.harness import build_records, index_corpus

pytestmark = pytest.mark.eval


async def test_ragas_metrics_are_computed() -> None:
    pytest.importorskip("ragas", reason="ragas (or its langchain stack) is not importable here")
    pytest.importorskip("langchain_openai", reason="langchain-openai is not importable here")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("ragas evaluation needs OPENAI_API_KEY")

    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings as LangchainEmbeddings
    from openai import OpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    from knowledge_synthesizer.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
    from knowledge_synthesizer.adapters.llm.openai_llm import OpenAILLM
    from knowledge_synthesizer.adapters.retrieval.query_transformers import MultiQueryTransformer
    from knowledge_synthesizer.adapters.retrieval.vector_retriever import VectorRetriever
    from knowledge_synthesizer.application.qa import QAService
    from tests.fakes import InMemoryVectorStore

    client = OpenAI()
    embeddings = OpenAIEmbeddings(client)
    store = InMemoryVectorStore()
    index_corpus(embeddings, store, EVAL_CORPUS)

    llm = OpenAILLM(client)
    retriever = VectorRetriever(embeddings, store, MultiQueryTransformer(llm))
    qa = QAService(retriever, llm, k=3)

    records = await build_records(qa, retriever, EVAL_SAMPLES, k=3)
    dataset = EvaluationDataset.from_list(records)

    judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    judge_embeddings = LangchainEmbeddingsWrapper(
        LangchainEmbeddings(model="text-embedding-3-small")
    )
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=judge,
        embeddings=judge_embeddings,
    )

    scores = result.to_pandas()
    assert len(scores) == len(EVAL_SAMPLES)
    assert "faithfulness" in result.scores[0]
