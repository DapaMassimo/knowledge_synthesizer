# Knowledge Synthesizer

Ingest heterogeneous sources (web pages, PDF, PPTX, DOCX, plain text) and, for a chosen
topic, produce **(1)** a structured summary of the key points and **(2)** a
question-answering interface grounded in those sources, with citations. It is a
retrieval-augmented generation (RAG) system plus a summarization pipeline sharing one
indexed corpus.

## Architecture

Hexagonal (ports & adapters), with a strict inward-only dependency rule:

- `domain` — entities, value objects, and ports (`Protocol`). No framework deps.
- `application` — use cases orchestrating the ports. Depends on `domain` only.
- `adapters` — concrete port implementations (loaders, parser, embeddings, vector store,
  LLM, retrievers).
- `composition` — the composition root that wires adapters into use cases.
- `entrypoints` — CLI and Streamlit UI.

The dependency rule is enforced by **import-linter** (`make lint`).

## Quick start

```bash
make install   # uv sync --all-groups
make check     # lint + import-linter + mypy (strict) + tests with coverage
make test      # tests only (excludes eval)
make run       # Streamlit UI (added in a later phase)
make cli       # CLI entrypoint (added in a later phase)
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Copy `.env.example` to `.env`
and fill in secrets when you reach the phases that need them.
