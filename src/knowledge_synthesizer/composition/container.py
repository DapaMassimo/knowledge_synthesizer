"""Composition root: build use cases from settings by wiring concrete adapters.

This is the only module allowed to import concrete adapters.
"""

from __future__ import annotations

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


class Container:
    """Lazily builds and wires use cases. Shared infra (client, store, cache) is memoized."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: OpenAI | None = None
        self._store: ChromaVectorStore | None = None
        self._cache: DoclingDocumentCache | None = None

    def indexing_service(self) -> IndexingService:
        cache = self._docling_cache()
        artifacts = self._settings.docling_artifacts_path
        return IndexingService(
            loader=RoutingSourceLoader(FileLoader(), WebLoader()),
            parser=DoclingParser(
                cache=cache,
                use_cache=self._settings.use_cache,
                artifacts_path=Path(artifacts) if artifacts else None,
                do_ocr=self._settings.docling_do_ocr,
            ),
            chunker=DoclingChunker(cache=cache),
            embeddings=self._embeddings(),
            store=self._vector_store(),
        )

    def qa_service(self) -> QAService:
        return QAService(self._retriever(), self._llm(), k=self._settings.retriever_k)

    def summarization_service(self) -> SummarizationService:
        return SummarizationService(
            self._retriever(),
            MapReduceSummarizer(self._llm()),
            k=self._settings.summary_k,
        )

    def _openai(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self._settings.openai_api_key or None)
        return self._client

    def _embeddings(self) -> EmbeddingModel:
        return OpenAIEmbeddings(self._openai(), model=self._settings.embedding_model)

    def _llm(self) -> LLMProvider:
        return OpenAILLM(self._openai(), model=self._settings.llm_model)

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

    def _transformer(self) -> QueryTransformer:
        if self._settings.query_strategy == "multiquery":
            return MultiQueryTransformer(self._llm())
        if self._settings.query_strategy == "hyde":
            return HydeTransformer(self._llm())
        return PassthroughTransformer()

    def _retriever(self) -> Retriever:
        return VectorRetriever(self._embeddings(), self._vector_store(), self._transformer())
