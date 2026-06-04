# Knowledge Synthesizer

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Lint](https://img.shields.io/badge/lint-ruff-46a2f1)
![Types](https://img.shields.io/badge/types-mypy%20strict-2a6db0)
![Tests](https://img.shields.io/badge/tests-pytest-0a9edc)
![License](https://img.shields.io/badge/license-MIT-green)

Ingest heterogeneous sources — PDFs, slides, Word docs, plain text, and web pages — and turn
them into **cited answers** and **structured summaries**. A retrieval-augmented generation
(RAG) system plus a summarization pipeline over one shared, persistent index.

---

## Features

- 📥 **Heterogeneous ingestion** — PDF, PPTX, DOCX, TXT, Markdown and web URLs, parsed with [Docling](https://github.com/docling-project/docling).
- 🔎 **Grounded QA with citations** — every answer cites the source file, page and section; web sources render as links.
- 🧭 **Query transformation** — multi-query (default), HyDE, or plain similarity, behind a swappable port.
- 🧱 **Structured summaries** — map-reduce summarization of a topic across the whole corpus.
- ♻️ **Idempotent indexing** — re-indexing the same document is skipped (content-hash dedup, persisted across restarts).
- 🖥️ **Streamlit UI** — drag-and-drop upload, per-document OCR, chat history with copy & download, a live model picker, one-click document removal, plus configuration and live log panels.
- ⌨️ **CLI** — `index`, `ask`, `summarize`.
- 🧩 **Offline-friendly parsing** — local Docling models, an OCR toggle, and a content-hash parse cache.

## Architecture

Hexagonal (ports & adapters), with an inward-only dependency rule **enforced by import-linter**:

```
entrypoints (CLI · Streamlit)
   │  calls
composition (DI container — the only place that imports concrete adapters)
   │  wires
adapters  ──────────────►  domain  ◄──────────────  application
(Docling, Chroma, OpenAI,  (entities + Protocol      (use cases: indexing,
 httpx + trafilatura)       ports, no frameworks)      QA, summarization)
```

- **domain** — Pydantic entities + `Protocol` ports. Zero framework dependencies.
- **application** — use cases orchestrating ports only (framework-free).
- **adapters** — concrete implementations (loaders, parser, embeddings, vector store, LLM, retrievers).
- **composition** — builds and wires everything from settings.
- **entrypoints** — CLI and Streamlit.

**Pipeline:** `load → parse → chunk → embed → store`, then `transform query → retrieve → generate → cite`.

## Tech stack

Python 3.12+ · [uv](https://docs.astral.sh/uv/) · Pydantic v2 · Docling · Chroma · OpenAI
(`gpt-4o-mini` + `text-embedding-3-small`) · httpx + trafilatura · Streamlit · ruff · mypy
(strict) · pytest · import-linter.

## Quick start

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), and an OpenAI API key.

```bash
make install                 # uv sync --all-groups
cp .env.example .env          # then set OPENAI_API_KEY (or export it)
make run                      # Streamlit UI → http://localhost:8501
```

In the UI: **add sources → Index → ask in the _Ask_ tab** (or get an overview in _Summary_).

### CLI

```bash
uv run knowledge-synthesizer index report.pdf https://example.com/article
uv run knowledge-synthesizer ask "What are the key risks?"
uv run knowledge-synthesizer summarize "deployment architecture"
```

The index is persistent (`.chroma/`), so `index` and `ask`/`summarize` work across separate runs.

### Offline / faster PDF parsing (optional)

Pre-download the Docling models once, then point the app at them:

```bash
uv run python -c "from docling.utils.model_downloader import download_models; download_models()"
```
```dotenv
KS_DOCLING_ARTIFACTS_PATH=~/.cache/docling/models
KS_DOCLING_DO_OCR=false        # OCR only helps for *scanned* PDFs
```

## Configuration

Everything is configured via environment / `.env` (prefix `KS_`; the OpenAI key uses its
conventional name). The most useful knobs:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | – | OpenAI key (LLM **and** embeddings) |
| `KS_LLM_MODEL` | `gpt-4o-mini` | chat model for answers & summaries |
| `KS_EMBEDDING_MODEL` | `text-embedding-3-small` | embedding model (don't change without re-indexing) |
| `KS_QUERY_STRATEGY` | `multiquery` | `multiquery` \| `hyde` \| `passthrough` |
| `KS_RETRIEVER_K` | `8` | chunks retrieved per query |
| `KS_CHROMA_PATH` | `.chroma` | persistent index dir (empty → in-memory) |
| `KS_DOCLING_DO_OCR` | `true` | run OCR during PDF parsing |
| `KS_DOCLING_ARTIFACTS_PATH` | – | local Docling models dir (offline parsing) |
| `KS_LLM_MODELS` / `KS_EMBEDDING_MODELS` | curated list | comma-separated options for the UI dropdown |
| `KS_LOG_LEVEL` | `INFO` | application log level |

See [`.env.example`](.env.example) for the full list.

## Development

```bash
make check        # ruff + import-linter + mypy (strict) + pytest with coverage  ← the gate
make test         # tests only (excludes the opt-in ragas eval)
make format       # ruff format
make reset        # stop the app and wipe .chroma / .uploads / the parse cache
```

- The hexagonal dependency rule is enforced by **import-linter** (`domain`/`application` import no adapters or IO/orchestration frameworks).
- The ragas evaluation harness is **opt-in**: `make test-eval`.

## Project layout

```
src/knowledge_synthesizer/
├── domain/        # entities, ports (Protocols), errors
├── application/   # indexing, qa, summarization use cases
├── adapters/      # loaders, parsing (Docling), embeddings, vectorstores (Chroma), llm, retrieval
├── composition/   # DI container (wires use cases from settings)
├── config/        # pydantic-settings
└── entrypoints/   # cli.py, streamlit_app.py, presentation helpers
tests/             # unit, integration, eval
```

## Notes

- **Cost** — built on `gpt-4o-mini` + `text-embedding-3-small`; a typical session costs a fraction of a cent. Parsing, chunking and the vector store run **locally** (no API cost).
- **OCR** — keep it **off** for digital PDFs (text is already extractable); turn it **on** only for scanned documents. It's a per-document choice in the UI.
- **Web scraping** — direct fetch (`httpx`) + main-content extraction (`trafilatura`); no third-party search API.

## License

MIT.
