"""Streamlit UI. Adds sources, indexes them, shows a topic summary, and chat-style QA.

The UI calls the container only; all logic lives in the use cases. Pure formatting lives in
``presentation`` so it can be unit-tested without a Streamlit runtime.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import streamlit as st

from knowledge_synthesizer.composition.container import Container
from knowledge_synthesizer.config.settings import Settings
from knowledge_synthesizer.entrypoints.logsetup import clear_logs, configure_logging, get_logs
from knowledge_synthesizer.entrypoints.presentation import (
    answer_markdown,
    mask_secret,
    materialize_uploads,
    parse_sources,
    summary_markdown,
)

_UPLOAD_DIR = Path(".uploads")
_UPLOAD_TYPES = ["pdf", "pptx", "docx", "txt", "md", "html"]

_STEPS = (
    "#### How to use\n"
    "1. **Add sources** — upload documents (or paste file paths / URLs) in the left sidebar.\n"
    "2. **Click _Index_** — process them into the searchable store. Do this once per document"
    " (re-indexing the same file is skipped); large PDFs take a little while.\n"
    "3. **Summary tab** — type a topic and click _Summarize_ for a structured overview.\n"
    "4. **Ask tab** — ask questions and get answers with inline citations.\n\n"
    "_Summarize and Ask only use what you've already indexed._"
)


@st.cache_resource
def _container() -> Container:
    return Container(Settings())


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _render() -> None:
    st.set_page_config(page_title="Knowledge Synthesizer", layout="wide")
    settings = Settings()
    if configure_logging(settings.log_level):
        logging.getLogger("knowledge_synthesizer.entrypoints").info(
            "Startup — OpenAI key %s · model %s · embeddings %s · strategy %s · k=%d",
            mask_secret(settings.openai_api_key),
            settings.llm_model,
            settings.embedding_model,
            settings.query_strategy,
            settings.retriever_k,
        )

    st.title("Knowledge Synthesizer")
    st.caption(
        f"🔑 OpenAI key: `{mask_secret(settings.openai_api_key)}`  ·  "
        f"model `{settings.llm_model}`  ·  strategy `{settings.query_strategy}` "
        f"(k={settings.retriever_k})"
    )
    st.markdown(_STEPS)
    container = _container()

    with st.sidebar:
        st.header("Sources")
        uploads = st.file_uploader(
            "Upload documents",
            type=_UPLOAD_TYPES,
            accept_multiple_files=True,
            key="uploads_input",
        )
        raw_sources = st.text_area(
            "…or paste file paths / URLs (one per line)", key="sources_input"
        )
        if st.button("Index", key="index_button"):
            sources = parse_sources(raw_sources)
            if uploads:
                sources += materialize_uploads(
                    [(file.name, file.getvalue()) for file in uploads], _UPLOAD_DIR
                )
            if not sources:
                st.warning("Upload a document or add a file path / URL.")
            else:
                with st.spinner("Indexing…"):
                    report = _run(container.indexing_service().index(sources))
                st.success(
                    f"Indexed {report.documents_indexed} document(s), "
                    f"skipped {report.documents_skipped}, {report.chunks_indexed} chunk(s)."
                )

    summary_tab, qa_tab = st.tabs(["Summary", "Ask"])

    with summary_tab:
        topic = st.text_input("Topic", key="topic_input")
        if st.button("Summarize", key="summarize_button") and topic.strip():
            with st.spinner("Summarizing…"):
                summary = _run(container.summarization_service().summarize(topic.strip()))
            st.markdown(summary_markdown(summary))

    with qa_tab:
        question = st.chat_input("Ask a question about your sources")
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"), st.spinner("Thinking…"):
                answer = _run(container.qa_service().ask(question))
                st.markdown(answer_markdown(answer))

    with st.expander("📋 Logs", expanded=False):
        if st.button("Clear logs", key="clear_logs_button"):
            clear_logs()
        lines = get_logs()
        st.code("\n".join(lines[-200:]) if lines else "(no logs yet)", language="log")


_render()
