"""Streamlit UI. Adds sources, indexes them, shows a topic summary, and chat-style QA.

The UI calls the container only; all logic lives in the use cases. Pure formatting lives in
``presentation`` so it can be unit-tested without a Streamlit runtime.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

import streamlit as st

from knowledge_synthesizer.composition.container import Container
from knowledge_synthesizer.config.settings import Settings
from knowledge_synthesizer.entrypoints.presentation import (
    answer_markdown,
    parse_sources,
    summary_markdown,
)


@st.cache_resource
def _container() -> Container:
    return Container(Settings())


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _render() -> None:
    st.set_page_config(page_title="Knowledge Synthesizer", layout="wide")
    st.title("Knowledge Synthesizer")
    container = _container()

    with st.sidebar:
        st.header("Sources")
        raw_sources = st.text_area("File paths or URLs (one per line)", key="sources_input")
        if st.button("Index", key="index_button"):
            sources = parse_sources(raw_sources)
            if not sources:
                st.warning("Add at least one file path or URL.")
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


_render()
