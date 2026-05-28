"""Application configuration via pydantic-settings (env / .env)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KS_", env_file=".env", extra="ignore", populate_by_name=True
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_backend: Literal["openai"] = "openai"

    chroma_path: str | None = ".chroma"
    collection_name: str = "knowledge"
    docling_cache_dir: str | None = ".cache/docling"
    docling_artifacts_path: str | None = None
    docling_do_ocr: bool = True
    use_cache: bool = True

    query_strategy: Literal["multiquery", "hyde", "passthrough"] = "multiquery"
    retriever_k: int = 5
    summary_k: int = 12
