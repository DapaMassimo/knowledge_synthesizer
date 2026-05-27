import pytest

from knowledge_synthesizer.adapters.retrieval.query_transformers import PassthroughTransformer
from knowledge_synthesizer.adapters.retrieval.vector_retriever import VectorRetriever
from knowledge_synthesizer.application.qa import QAService
from tests.eval.dataset import EVAL_CORPUS, EVAL_SAMPLES
from tests.eval.harness import build_records, index_corpus
from tests.fakes import FakeEmbeddings, FakeLLM, InMemoryVectorStore

pytestmark = pytest.mark.unit


def test_index_corpus_populates_store() -> None:
    embeddings = FakeEmbeddings(dim=64)
    store = InMemoryVectorStore()
    index_corpus(embeddings, store, EVAL_CORPUS)
    assert len(store) == len(EVAL_CORPUS)


async def test_build_records_has_ragas_shape() -> None:
    embeddings = FakeEmbeddings(dim=64)
    store = InMemoryVectorStore()
    index_corpus(embeddings, store, EVAL_CORPUS)
    retriever = VectorRetriever(embeddings, store, PassthroughTransformer())
    qa = QAService(retriever, FakeLLM(response="Rome is the capital of Italy [1]."), k=2)

    records = await build_records(qa, retriever, EVAL_SAMPLES, k=2)

    assert len(records) == len(EVAL_SAMPLES)
    for record, sample in zip(records, EVAL_SAMPLES, strict=True):
        assert record["user_input"] == sample.question
        assert record["reference"] == sample.ground_truth
        assert isinstance(record["response"], str)
        contexts = record["retrieved_contexts"]
        assert isinstance(contexts, list)
        assert contexts  # non-empty
