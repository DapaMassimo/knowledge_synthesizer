"""Composition root: build use cases from settings by wiring concrete adapters.

This is the only module allowed to import concrete adapters.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path

from openai import OpenAI

from knowledge_synthesizer.adapters.embeddings.openai_embeddings import OpenAIEmbeddings
from knowledge_synthesizer.adapters.llm.openai_llm import OpenAILLM
from knowledge_synthesizer.adapters.loaders.file_loader import FileLoader
from knowledge_synthesizer.adapters.loaders.routing_loader import RoutingSourceLoader
from knowledge_synthesizer.adapters.loaders.web_loader import WebLoader
from knowledge_synthesizer.adapters.parsing.docling_cache import DoclingDocumentCache
from knowledge_synthesizer.adapters.parsing.docling_chunker import DoclingChunker
from knowledge_synthesizer.adapters.parsing.docling_parser import DoclingParser
from knowledge_synthesizer.adapters.retrieval.query_transformers import (
    HydeTransformer,
    MultiQueryTransformer,
    PassthroughTransformer,
)
from knowledge_synthesizer.adapters.retrieval.vector_retriever import VectorRetriever
from knowledge_synthesizer.adapters.vectorstores.chroma_store import ChromaVectorStore
from knowledge_synthesizer.application.indexing import IndexingService
from knowledge_synthesizer.application.qa import QAService
from knowledge_synthesizer.application.summarization import (
    MapReduceSummarizer,
    SummarizationService,
)
from knowledge_synthesizer.config.settings import Settings
from knowledge_synthesizer.domain.ports import (
    EmbeddingModel,
    LLMProvider,
    QueryTransformer,
    Retriever,
    VectorStore,
)


def _create_openai_client(api_key: str | None) -> OpenAI:
    try:
        return OpenAI(api_key=api_key)
    except ssl.SSLError:
        # python-build-standalone's bundled OpenSSL can fail to initialize a mismatched
        # system openssl.cnf ("CONF MODULE_INITIALIZATION_ERROR"). Bypass the OpenSSL config
        # file and retry; on healthy systems the first attempt succeeds and config is kept.
        os.environ["OPENSSL_CONF"] = os.devnull
        return OpenAI(api_key=api_key)


class Container:
    """Lazily builds and wires use cases. Shared infra (client, store, cache) is memoized."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: OpenAI | None = None
        self._store: ChromaVectorStore | None = None
        self._cache: DoclingDocumentCache | None = None

    def indexing_service(
        self, *, do_ocr: bool | None = None, embedding_model: str | None = None
    ) -> IndexingService:
        cache = self._docling_cache()
        artifacts = self._settings.docling_artifacts_path
        ocr = self._settings.docling_do_ocr if do_ocr is None else do_ocr
        return IndexingService(
            loader=RoutingSourceLoader(FileLoader(), WebLoader()),
            parser=DoclingParser(
                cache=cache,
                use_cache=self._settings.use_cache,
                artifacts_path=Path(artifacts) if artifacts else None,
                do_ocr=ocr,
            ),
            chunker=DoclingChunker(cache=cache),
            embeddings=self._embeddings(embedding_model),
            store=self._vector_store(),
        )

    def qa_service(
        self, *, llm_model: str | None = None, embedding_model: str | None = None
    ) -> QAService:
        retriever = self._retriever(embedding_model=embedding_model, llm_model=llm_model)
        return QAService(retriever, self._llm(llm_model), k=self._settings.retriever_k)

    def summarization_service(
        self, *, llm_model: str | None = None, embedding_model: str | None = None
    ) -> SummarizationService:
        retriever = self._retriever(embedding_model=embedding_model, llm_model=llm_model)
        return SummarizationService(
            retriever,
            MapReduceSummarizer(self._llm(llm_model)),
            k=self._settings.summary_k,
        )

    def indexed_sources(self) -> list[str]:
        """Source URIs already in the vector store (persisted across restarts)."""
        return self._vector_store().indexed_sources()

    def remove_source(self, source_uri: str) -> None:
        """Delete a document and its embeddings from the vector store."""
        self._vector_store().delete_source(source_uri)

    def _openai(self) -> OpenAI:
        if self._client is None:
            self._client = _create_openai_client(self._settings.openai_api_key or None)
        return self._client

    def _embeddings(self, model: str | None = None) -> EmbeddingModel:
        return OpenAIEmbeddings(self._openai(), model=model or self._settings.embedding_model)

    def _llm(self, model: str | None = None) -> LLMProvider:
        return OpenAILLM(self._openai(), model=model or self._settings.llm_model)

    def _vector_store(self) -> VectorStore:
        if self._store is None:
            if self._settings.chroma_path:
                self._store = ChromaVectorStore.persistent(
                    self._settings.chroma_path, self._settings.collection_name
                )
            else:
                self._store = ChromaVectorStore.ephemeral(self._settings.collection_name)
        return self._store

    def _docling_cache(self) -> DoclingDocumentCache:
        if self._cache is None:
            cache_dir = (
                Path(self._settings.docling_cache_dir) if self._settings.docling_cache_dir else None
            )
            self._cache = DoclingDocumentCache(cache_dir=cache_dir)
        return self._cache

    def _transformer(self, llm_model: str | None = None) -> QueryTransformer:
        if self._settings.query_strategy == "multiquery":
            return MultiQueryTransformer(self._llm(llm_model))
        if self._settings.query_strategy == "hyde":
            return HydeTransformer(self._llm(llm_model))
        return PassthroughTransformer()

    def _retriever(
        self, *, embedding_model: str | None = None, llm_model: str | None = None
    ) -> Retriever:
        return VectorRetriever(
            self._embeddings(embedding_model),
            self._vector_store(),
            self._transformer(llm_model),
        )
