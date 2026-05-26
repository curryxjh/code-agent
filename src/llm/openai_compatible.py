from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterable

from openai import AsyncOpenAI
from src.llm.types import ChatOptions, ChatResponse, Message, StreamEvent, TextBlock, ToolUseBlock, ToolResultBlock


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
            if isinstance(m.content, str):
                formatted.append({"role": m.role, "content": m.content})
                continue
            # ContentBlock list
            if m.role == "assistant":
                text_parts = [b for b  in m.content if b.type == "text"]
                tool_uses = [b for b in m.content if b.type == "tool_use"]
                msg: dict[str, str | None | list[dict]] = {
                    "role": m.role,
                    "content": "".join(b.text for b in text_parts) if text_parts else None,
                }
                if tool_uses:
                    msg["tool_calls"] = [
                        {
                            "id": t.id,
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "arguments": json.dumps(t.input),
                            },
                        }
                        for t in tool_uses
                    ]
                formatted.append(msg)
            else:
                for block in m.content:
                    if block.type == "tool_result":
                        formatted.append({
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.content,
                        })
        return formatted

    async def chat(
        self, messages: list[Message], options: ChatOptions | None = None
    ) -> ChatResponse:
        options = options or ChatOptions()

        create_params: dict = {
            "model": self._model,
            "max_tokens": options.max_tokens or 4096,
            "messages": self._format_messages(messages, options.system),
        }

        if options.tools:
            create_params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in options.tools
            ]

        response = await self._client.chat.completions.create(**create_params)

        choice = response.choices[0]

        content_blocks = []
        if choice.message.content:
            content_blocks.append(TextBlock(text=choice.message.content))
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_blocks.append(
                    ToolUseBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments),
                    )
                )

        if choice.finish_reason == "stop":
            stop_reason = "end_turn"
        elif choice.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "max_tokens"

        return ChatResponse(
            content=content_blocks,
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