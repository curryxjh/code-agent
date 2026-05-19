from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, AsyncIterable


@dataclass
class TextBlock:
    """Text content block."""
    type: str = "text"
    text: str = ""

@dataclass
class ToolUseBlock:
    """Tool use block."""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)

@dataclass
class ToolResultBlock:
    """Tool result block."""
    type: str = "tool_result"
    tool_use_id: str = ""
    content: str = ""
    is_error: bool = False

ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock

@dataclass
class Tool:
    """Tool definition for LLM."""
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)


@dataclass
class StreamEvent:
    """Event emitted by LLM when a message is received."""
    type: str # "message_start", "text_delta", "message_stop", "error"
    text: str | None = None

@dataclass
class Message:
    "Core message type for LLM conversations."
    role: str
    content: str | list[ContentBlock] = ""

@dataclass
class ChatResponse:
    """Response from a chat completion."""
    content: list[ContentBlock] = field(default_factory=list)
    text: str = ""
    stop_reason: str = "" # "end_turn", "max_tokens", or "tool_use"
    usage: dict = field(default_factory=dict)

@dataclass
class ChatOptions:
    """Options for a chat completion."""
    system: str | None = None
    max_tokens: int | None = None
    tools: list[Tool] | None = None

class LLMProvider(Protocol):
    """Unified interface for LLM providers."""

    async def chat(self, messages: list[Message], options: ChatOptions | None = None) -> ChatResponse:
        ...

    async def stream(self, messages: list[Message], options: ChatOptions | None = None) -> AsyncIterable[StreamEvent]:
        ...