from __future__ import annotations
from dataclasses import dataclass
from openai import AsyncOpenAI
from .types import ChatOptions, ChatResponse, Message


@dataclass
class OpenAICompatibleConfig:
    api_key: str
    base_url: str
    model: str


class OpenAICompatibleProvider:
    """LLM provider for OpenAI-compatible APIs (DeepSeek, Qwen, etc.)."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self._client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self._model = config.model

    def _format_message(self, messages: list[Message], system: str | None) -> list[dict]:
        """Format messages for OpenAI API, prepending system message if present."""