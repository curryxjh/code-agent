from __future__ import annotations

import math
import json
from dataclasses import dataclass
from src.llm import (
    ContentBlock, TextBlock, ToolUseBlock, ToolResultBlock, Message, Tool
)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a string.

    Uses a simple heuristic: ~4 characters per token for English,
    ~2 characters per token for CJK text. This is faster than calling
    the tokenizer API and sufficient for budget management.
    """
    if not text:
        return 0

    cjk_chars = 0
    other_chars = 0

    for ch in text:
        code = ord(ch)
        # CJK Unified Ideographs + common ranges
        if (
                (0x4E00 <= code <= 0x9FFF)  # CJK Unified
                or (0x3000 <= code <= 0x303F)  # CJK Punctuation
                or (0x3040 <= code <= 0x30FF)  # Hiragana + Katakana
                or (0xFF00 <= code <= 0xFFEF)  # Fullwidth forms
        ):
            cjk_chars += 1
        else:
            other_chars += 1

    # ~2 chars/token for CJK, ~4 chars/token for other
    return math.ceil(cjk_chars / 2) + math.ceil(other_chars / 4)

def _estimate_block_tokens(block: ContentBlock) -> int:
    """Estimate tokens in a single content block."""
    if isinstance(block, TextBlock):
        return estimate_tokens(block.text)
    elif isinstance(block, ToolUseBlock):
        return estimate_tokens(block.name) + estimate_tokens(json.dumps(block.input))
    elif isinstance(block, ToolResultBlock):
        return estimate_tokens(block.content)
    return 0

def estimate_message_tokens(message: Message) -> int:
    """Estimate token count for a single message."""
    # Base overhead per message (role tag, formatting)
    overhead = 4

    if isinstance(message.content, str):
        return overhead + estimate_tokens(message.content)

        # Array of content blocks
    return overhead + sum(_estimate_block_tokens(b) for b in message.content)

def estimate_conversation_tokens(
    messages: list[Message],
    system: str | None = None,
    tools: list[Tool] | None = None,
) -> int:
    """Estimate total tokens for a conversation (messages + system + tools)."""
    total = 0

    # System prompt
    if system:
        total += estimate_tokens(system)

    # Tool definitions
    if tools:
        for tool in tools:
            total += (
                estimate_tokens(tool.name)
                + estimate_tokens(tool.description)
                + estimate_tokens(json.dumps(tool.input_schema))
            )

    # Messages
    for msg in messages:
        total += estimate_message_tokens(msg)

    return total

# Known context window limits for common models
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-haiku-4-20250414": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "deepseek-chat": 64_000,
    "deepseek-coder": 128_000,
}


def get_model_context_limit(model: str) -> int | None:
    """Get the context window limit for a model. Returns None if unknown."""
    return MODEL_CONTEXT_LIMITS.get(model)


@dataclass
class ContextBudget:
    """Context window budget manager."""

    max_context_tokens: int = 64_000
    reserved_for_response: int = 4096

DEFAULT_BUDGET = ContextBudget()


def remaining_budget(budget: ContextBudget, used_tokens: int) -> int:
    """Calculate remaining budget for input tokens."""
    return max(0, budget.max_context_tokens - budget.reserved_for_response - used_tokens)


def is_over_budget(budget: ContextBudget, used_tokens: int) -> bool:
    """Check if adding more tokens would exceed the budget."""
    return used_tokens >= budget.max_context_tokens - budget.reserved_for_response