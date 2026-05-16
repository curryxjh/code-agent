from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterable

from openai import AsyncOpenAI
from src.llm.types import ChatOptions, ChatResponse, Message, StreamEvent


@dataclass
class OpenAICompatibleConfig:
    api_key: str
    base_url: str
    model: str


class OpenAICompatibleProvider:
    """LLM provider for OpenAI-compatible APIs (DeepSeek, Qwen, etc.)."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self._client = AsyncOpenAI(
            api_key=config.api_key, base_url=config.base_url
        )
        self._model = config.model

    def _format_messages(
        self, messages: list[Message], system: str | None = None
    ) -> list[dict]:
        """Format messages for OpenAI API, prepending system message if present."""
        formatted: list[dict] = []
        if system:
            formatted.append({"role": "system", "content": system})
        for m in messages:
            formatted.append({"role": m.role, "content": m.content})
        return formatted

    async def chat(
        self, messages: list[Message], options: ChatOptions | None = None
    ) -> ChatResponse:
        options = options or ChatOptions()

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=options.max_tokens or 4096,
            messages=self._format_messages(messages, options.system),
        )

        choice = response.choices[0]
        stop_reason = (
            "end_turn" if choice.finish_reason == "stop" else "max_tokens"
        )

        return ChatResponse(
            text=choice.message.content or "",
            stop_reason=stop_reason,
            usage={
                "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                "output_tokens": getattr(
                    response.usage, "completion_tokens", 0
                ),
            },
        )

    async def stream(self, messages: list[Message], options: ChatOptions | None = None) -> AsyncIterable[StreamEvent]:
        options = options or ChatOptions()

        response = await self._client.chat.completions.create(
            model=self._model,
            max_tokens=options.max_tokens or 4096,
            messages=self._format_messages(messages, options.system),
            stream=True,
        )

        yield StreamEvent(type="message_start")

        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield StreamEvent(type="text_delta", text=delta.content)

        yield StreamEvent(type="message_stop")