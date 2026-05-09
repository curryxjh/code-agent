from __future__ import annotations
from dataclasses import dataclass
from anthropic import AsyncAnthropic
from .types import ChatOptions, ChatResponse, Message

@dataclass
class AnthropicConfig:
    api_key: str
    model: str = "claude-sonnet-4-20250514"

class AnthropicProvider:
    """LLM provider for Anthropic Claude models."""

    def __init__(self, config: AnthropicConfig) -> None:
        self._client = AsyncAnthropic(api_key=config.api_key)
        self._model = config.model

    async def chat(self, messages: list[Message], options: ChatOptions | None = None) -> ChatResponse:
        options = options or ChatOptions()

        params: dict = {
            "model": self._model,
            "max_tokens": options.max_tokens or 4096,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages
            ]
        }

        if options.system:
            params["system"] = options.system

        response = await self._client.messages.create(**params)

        text = "".join(
            b.text for b in response.content if b.type == "text"
        )

        stop_reason = (
            "end_turn" if response.stop_reason == "end_turn" else "max_tokens"
        )

        return ChatResponse(
            text=text,
            stop_reason=stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )