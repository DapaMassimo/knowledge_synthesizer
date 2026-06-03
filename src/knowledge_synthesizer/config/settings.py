"""Application configuration via pydantic-settings (env / .env)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KS_", env_file=".env", extra="ignore", populate_by_name=True
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_backend: Literal["openai"] = "openai"

    # Selectable models in the UI dropdowns (comma-separated in env, e.g. KS_LLM_MODELS=a,b,c).
    llm_models: Annotated[list[str], NoDecode] = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ]
    embedding_models: Annotated[list[str], NoDecode] = [
        "text-embedding-3-small",
        "text-embedding-3-large",
    ]

    @field_validator("llm_models", "embedding_models", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    chroma_path: str | None = ".chroma"
    collection_name: str = "knowledge"
    docling_cache_dir: str | None = ".cache/docling"
    docling_artifacts_path: str | None = None
    docling_do_ocr: bool = True
    use_cache: bool = True

    query_strategy: Literal["multiquery", "hyde", "passthrough"] = "multiquery"
    retriever_k: int = 8
    summary_k: int = 12

    log_level: str = "INFO"
