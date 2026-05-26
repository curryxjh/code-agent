from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterable

from anthropic import AsyncAnthropic
from openai.resources.containers.files import content

from src.llm.types import (
    ChatOptions, ChatResponse, Message, StreamEvent,
    TextBlock, ToolUseBlock, ToolResultBlock, ContentBlock
)


@dataclass
class AnthropicConfig:
    api_key: str
    model: str = "claude-sonnet-4-20250514"

class AnthropicProvider:
    """LLM provider for Anthropic Claude models."""

    def __init__(self, config: AnthropicConfig) -> None:
        self._client = AsyncAnthropic(api_key=config.api_key)
        self._model = config.model

    @staticmethod
    def _format_content(content) -> str | list[dict]:
        """Convert content to Anthropic API format."""
        if isinstance(content, str):
            return content
        result = []
        for block in content:
            if block.type == "text":
                result.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                result.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
            elif block.type == "tool_result":
                d = {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": block.content,
                }
                if block.is_error:
                    d["is_error"] = True
                result.append(d)
        return result

    async def chat(self, messages: list[Message], options: ChatOptions | None = None) -> ChatResponse:
        options = options or ChatOptions()

        params: dict = {
            "model": self._model,
            "max_tokens": options.max_tokens or 4096,
            "messages": [
                {"role": m.role, "content": self._format_content(m.content)} for m in messages
            ]
        }

        if options.system:
            params["system"] = options.system

        if options.tools:
            params["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in options.tools
            ]

        response = await self._client.messages.create(**params)

        # Parse response content blocks
        content_blocks: list = []
        for b in response.content:
            if b.type == "tool_use":
                content_blocks.append(
                    ToolUseBlock(
                        id=b.id,
                        name=b.name,
                        input=b.input
                    )
                )
            else:
                content_blocks.append(TextBlock(text=b.text))

        text = "".join(b.text for b in content_blocks if b.type == "text")

        if response.stop_reason == "end_turn":
            stop_reason = "end_turn"
        elif response.stop_reason == "tool_use":
            stop_reason = "tool_use"
        else:
            stop_reason = "max_tokens"

        return ChatResponse(
            text=text,
            content=content_blocks,
            stop_reason=stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        )

    async def stream(self, messages: list[Message], options: ChatOptions | None = None) -> AsyncIterable[StreamEvent]:
        options = options or ChatOptions()

        params: dict = {
            "model": self._model,
            "max_tokens": options.max_tokens or 4096,
            "messages": [
                {"role": m.role, "content": self._format_content(m.content)} for m in messages
            ]
        }

        if options.system:
            params["system"] = options.system

        yield StreamEvent(type="message_start")

        async with self._client.messages.stream(**params) as stream:
            async for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    yield StreamEvent(type="text_delta", text=event.delta.text)

        yield StreamEvent(type="message_stop")

