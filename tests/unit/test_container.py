import pytest

from knowledge_synthesizer.adapters.retrieval.query_transformers import (
    HydeTransformer,
    MultiQueryTransformer,
    PassthroughTransformer,
)
from knowledge_synthesizer.application.indexing import IndexingService
from knowledge_synthesizer.application.qa import QAService
from knowledge_synthesizer.application.summarization import SummarizationService
from knowledge_synthesizer.composition.container import Container
from knowledge_synthesizer.config.settings import Settings
from knowledge_synthesizer.domain.models import Chunk, ChunkProvenance

pytestmark = pytest.mark.unit


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "openai_api_key": "test-key",
        "chroma_path": None,
        "docling_cache_dir": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def test_container_builds_use_cases() -> None:
    container = Container(_settings())
    assert isinstance(container.qa_service(), QAService)
    assert isinstance(container.summarization_service(), SummarizationService)
    assert isinstance(container.indexing_service(), IndexingService)


def test_vector_store_and_cache_are_memoized() -> None:
    container = Container(_settings())
    assert container._vector_store() is container._vector_store()
    assert container._docling_cache() is container._docling_cache()


def test_indexed_sources_is_empty_for_a_fresh_store() -> None:
    assert Container(_settings()).indexed_sources() == []


def test_remove_source_deletes_a_document_and_its_embeddings() -> None:
    container = Container(_settings())
    chunk = Chunk(
        chunk_id="a",
        text="x",
        document_hash="h",
        provenance=ChunkProvenance(source_uri="/a.pdf"),
    )
    container._vector_store().upsert([chunk], [[1.0, 0.0]])
    assert container.indexed_sources() == ["/a.pdf"]

    container.remove_source("/a.pdf")

    assert container.indexed_sources() == []


def test_openai_client_retries_after_ssl_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import ssl

    import knowledge_synthesizer.composition.container as module

    monkeypatch.setenv("OPENSSL_CONF", "placeholder")  # snapshot, auto-restored on teardown
    calls = {"n": 0}

    def flaky_openai(api_key: str | None = None) -> object:
        calls["n"] += 1
        if calls["n"] == 1:
            raise ssl.SSLError("CONF MODULE_INITIALIZATION_ERROR")
        return object()

    monkeypatch.setattr(module, "OpenAI", flaky_openai)

    client = module._create_openai_client("sk-test")

    assert calls["n"] == 2  # retried after the SSL error
    assert client is not None
    assert os.environ["OPENSSL_CONF"] == os.devnull


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("passthrough", PassthroughTransformer),
        ("hyde", HydeTransformer),
        ("multiquery", MultiQueryTransformer),
    ],
)
def test_query_strategy_selects_transformer(strategy: str, expected: type) -> None:
    container = Container(_settings(query_strategy=strategy))
    assert isinstance(container._transformer(), expected)


def test_model_overrides_are_applied() -> None:
    container = Container(_settings())
    assert container._llm()._model == "gpt-4o-mini"  # default from settings
    assert container._llm("gpt-4o")._model == "gpt-4o"  # runtime override
    assert container._embeddings("text-embedding-3-large")._model == "text-embedding-3-large"


def test_services_accept_runtime_overrides() -> None:
    container = Container(_settings())
    assert isinstance(container.qa_service(llm_model="gpt-4o"), QAService)
    assert isinstance(container.indexing_service(do_ocr=True), IndexingService)
    assert isinstance(
        container.summarization_service(embedding_model="text-embedding-3-large"),
        SummarizationService,
    )
