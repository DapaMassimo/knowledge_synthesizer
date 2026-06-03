"""List the models available to the OpenAI account, split into chat LLMs and embeddings."""

from __future__ import annotations

from openai import OpenAI

_LLM_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")
_EXCLUDE = (
    "embedding",
    "audio",
    "realtime",
    "transcribe",
    "tts",
    "image",
    "whisper",
    "dall-e",
    "moderation",
    "search",
)


def classify_models(model_ids: list[str]) -> tuple[list[str], list[str]]:
    """Partition model ids into (chat LLMs, embedding models), sorted."""
    llms: list[str] = []
    embeddings: list[str] = []
    for model_id in sorted(set(model_ids)):
        if model_id.startswith("text-embedding"):
            embeddings.append(model_id)
        elif model_id.startswith(_LLM_PREFIXES) and not any(
            token in model_id for token in _EXCLUDE
        ):
            llms.append(model_id)
    return llms, embeddings


def fetch_models(client: OpenAI) -> tuple[list[str], list[str]]:
    return classify_models([model.id for model in client.models.list()])
