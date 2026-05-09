from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Message:
    "Core message type for LLM conversations."
    role: str
    content: str

@dataclass
class ChatResponse:
    """Response from a chat completion."""
    text: str
    stop_reason: str # "end_turn" or "max_tokens"
    usage: dict = field(default_factory=dict)

@dataclass
class ChatOptions:
    """Options for a chat completion."""
    system: str | None = None
    max_tokens: int | None = None


class LLMProvider(Protocol):
    """Unified interface for LLM providers."""

    async def chat(self, messages: list[Message], options: ChatOptions | None = None) -> ChatResponse:
        ...