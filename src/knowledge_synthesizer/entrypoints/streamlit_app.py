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
from knowledge_synthesizer.domain.models import Answer, Source
from knowledge_synthesizer.entrypoints.logsetup import clear_logs, configure_logging, get_logs
from knowledge_synthesizer.entrypoints.presentation import (
    answer_markdown,
    conversation_markdown,
    mask_secret,
    materialize_uploads,
    model_options,
    parse_sources,
    settings_rows,
    summary_markdown,
)

_UPLOAD_TYPES = ["pdf", "pptx", "docx", "txt", "md", "html"]

_OCR_HELP = (
    "Default OFF. Turn ON only for **scanned** PDFs (pages that are images with no "
    "selectable text). For digital PDFs (selectable text) OCR adds a lot of time with no "
    "benefit. You can override it per document above."
)

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


def _index(container: Container, jobs: list[tuple[bool, Source]], embedding_model: str) -> str:
    """Index sources grouped by their OCR choice; returns a summary line."""
    by_ocr: dict[bool, list[Source]] = {}
    for do_ocr, source in jobs:
        by_ocr.setdefault(do_ocr, []).append(source)
    indexed = skipped = chunks = 0
    for do_ocr, sources in by_ocr.items():
        report = _run(
            container.indexing_service(do_ocr=do_ocr, embedding_model=embedding_model).index(
                sources
            )
        )
        indexed += report.documents_indexed
        skipped += report.documents_skipped
        chunks += report.chunks_indexed
    return f"Indexed {indexed} document(s), skipped {skipped}, {chunks} chunk(s)."


def _sidebar(container: Container, settings: Settings) -> tuple[str, str]:
    """Render the sidebar and return the selected (llm_model, embedding_model)."""
    uploads_dir = Path(settings.uploads_dir)
    llm_choices, embedding_choices = container.available_models()
    with st.sidebar:
        st.header("Sources")
        uploads = st.file_uploader(
            "Upload documents", type=_UPLOAD_TYPES, accept_multiple_files=True, key="uploads_input"
        )
        ocr_default = st.toggle("🔍 OCR (default)", value=settings.docling_do_ocr, help=_OCR_HELP)
        per_file_ocr: dict[str, bool] = {}
        if uploads:
            with st.expander("OCR per document", expanded=False):
                for file in uploads:
                    per_file_ocr[file.name] = st.checkbox(
                        file.name, value=ocr_default, key=f"ocr_{file.name}"
                    )
        raw_sources = st.text_area(
            "…or paste file paths / URLs (one per line)", key="sources_input"
        )

        llm_model = st.selectbox(
            "LLM model (answers & summaries)",
            model_options(settings.llm_model, [*settings.llm_models, *llm_choices]),
            help="Live list from your OpenAI account (falls back to KS_LLM_MODELS). "
            "Safe to change anytime — it doesn't affect the index.",
        )
        embedding_model = st.selectbox(
            "Embedding model (index & search)",
            model_options(
                settings.embedding_model, [*settings.embedding_models, *embedding_choices]
            ),
            help="⚠️ Changing this needs a fresh index (different vector space/size).",
        )
        if embedding_model != settings.embedding_model:
            st.warning("Embedding model changed: clear the index (rm -rf .chroma) and re-index.")

        if st.button("Index", key="index_button"):
            uploaded = (
                materialize_uploads([(f.name, f.getvalue()) for f in uploads], uploads_dir)
                if uploads
                else []
            )
            jobs: list[tuple[bool, Source]] = [
                (per_file_ocr.get(file.name, ocr_default), source)
                for file, source in zip(uploads or [], uploaded, strict=True)
            ]
            jobs += [(ocr_default, source) for source in parse_sources(raw_sources)]
            if not jobs:
                st.warning("Upload a document or add a file path / URL.")
            else:
                with st.spinner("Indexing…"):
                    st.success(_index(container, jobs, embedding_model))

        st.divider()
        try:
            indexed = container.indexed_sources()
        except Exception as exc:
            st.error(f"Could not read the index — restart the app to recover. ({exc})")
            indexed = []
        st.caption(f"**Indexed: {len(indexed)} document(s)** — kept across restarts")
        for source in indexed:
            label = source if source.startswith(("http://", "https://")) else Path(source).name
            row, action = st.columns([5, 1])
            row.caption(f"📄 {label}")
            if action.button("🗑️", key=f"remove_{source}", help="Remove this document & embeddings"):
                container.remove_source(source)
                st.rerun()

    return str(llm_model), str(embedding_model)


def _render() -> None:
    st.set_page_config(page_title="Knowledge Synthesizer", layout="wide")
    settings = Settings()
    rows = settings_rows(settings.model_dump())
    if configure_logging(settings.log_level):
        startup_log = logging.getLogger("knowledge_synthesizer.entrypoints")
        startup_log.info("Startup configuration (%d settings):", len(rows))
        for name, value in rows:
            startup_log.info("  %s = %s", name, value)

    st.title("Knowledge Synthesizer")
    st.caption(
        f"🔑 OpenAI key: `{mask_secret(settings.openai_api_key)}`  ·  "
        f"strategy `{settings.query_strategy}` (k={settings.retriever_k})"
    )
    with st.expander("⚙️ Configuration (from .env / defaults)", expanded=False):
        st.table({"setting": [name for name, _ in rows], "value": [value for _, value in rows]})
    st.markdown(_STEPS)

    container = _container()
    llm_model, embedding_model = _sidebar(container, settings)

    summary_tab, qa_tab = st.tabs(["Summary", "Ask"])

    with summary_tab:
        topic = st.text_input("Topic", key="topic_input")
        if st.button("Summarize", key="summarize_button") and topic.strip():
            with st.spinner("Summarizing…"):
                summary = _run(
                    container.summarization_service(
                        llm_model=llm_model, embedding_model=embedding_model
                    ).summarize(topic.strip())
                )
            st.markdown(summary_markdown(summary))

    with qa_tab:
        if "qa_history" not in st.session_state:
            st.session_state["qa_history"] = []
        history: list[Answer] = st.session_state["qa_history"]

        for past in history:
            with st.chat_message("user"):
                st.markdown(past.question)
            with st.chat_message("assistant"):
                st.markdown(answer_markdown(past))
                with st.popover("📋 Copy answer"):
                    st.code(past.text, language=None)

        if history:
            st.download_button(
                "⬇️ Download conversation",
                conversation_markdown(history),
                file_name="conversation.md",
                mime="text/markdown",
                key="download_conversation",
            )

        question = st.chat_input("Ask a question about your sources")
        if question:
            with st.spinner("Thinking…"):
                answer = _run(
                    container.qa_service(llm_model=llm_model, embedding_model=embedding_model).ask(
                        question
                    )
                )
            history.append(answer)
            st.rerun()

    with st.expander("📋 Logs", expanded=False):
        if st.button("Clear logs", key="clear_logs_button"):
            clear_logs()
        lines = get_logs()
        st.code("\n".join(lines[-200:]) if lines else "(no logs yet)", language="log")


_render()
