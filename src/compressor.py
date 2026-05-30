from dataclasses import dataclass
from src.llm.types import (
    Message, LLMProvider, TextBlock, ToolUseBlock, ToolResultBlock, ChatOptions
)
from src.token_counter import estimate_message_tokens

@dataclass
class CompressorConfig:
    """Configuration for conversation compression."""

    provider: LLMProvider
    max_tokens: int = 50000
    keep_recent_messages: int = 6
    summary_max_tokens: int = 1024

@dataclass
class CompressResult:
    """Result of a compression operation."""

    messages: list[Message]
    compressed: bool
    original_count: int
    compressed_count: int
    summary_tokens: int


def _format_content(message: Message) -> str:
    """Format a message's content for summarization."""
    if isinstance(message.content, str):
        return message.content

    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            parts.append(f"[Tool call: {block.name}]")
        elif isinstance(block, ToolResultBlock):
            parts.append(f"[Tool result: {block.content[:200]}]")
    return "\n".join(parts)


async def summarize_messages(
    provider: LLMProvider,
    messages: list[Message],
    max_tokens: int = 1024,
) -> str:
    """Generate a summary of messages using the LLM."""
    formatted = "\n\n".join(
        f"{m.role}: {_format_content(m)}" for m in messages
    )

    response = await provider.chat(
        [
            Message(
                role="user",
                content=(
                    "Summarize this conversation concisely. Focus on: what the user asked, "
                    "what tools were used, what was accomplished, and any important decisions "
                    f"or findings.\n\n{formatted}"
                ),
            ),
        ],
        options=ChatOptions(
            system=(
                "You are a conversation summarizer. Produce a concise summary that "
                "captures the key information needed to continue the conversation. "
                "Do not include pleasantries or meta-commentary."
            ),
            max_tokens=max_tokens,
        ),
    )

    return response.text



async def compress_conversation(
    config: CompressorConfig,
    messages: list[Message],
) -> CompressResult:
    """
    Compress a conversation by summarizing old messages and keeping recent ones.

    Strategy:
    1. Estimate total tokens in the conversation
    2. If under budget, return messages unchanged
    3. Otherwise, split into "old" and "recent" segments
    4. Summarize the old messages into a single summary message
    5. Return [summary, ...recent]
    """
    # Calculate total tokens
    total_tokens = sum(estimate_message_tokens(m) for m in messages)

    # Not over budget — no compression needed
    if total_tokens <= config.max_tokens or len(messages) <= config.keep_recent_messages:
        return CompressResult(
            messages=list(messages),
            compressed=False,
            original_count=len(messages),
            compressed_count=len(messages),
            summary_tokens=0,
        )

    # Split into old (to summarize) and recent (to keep)
    split_index = len(messages) - config.keep_recent_messages
    old_messages = messages[:split_index]
    recent_messages = messages[split_index:]

    # Summarize old messages
    summary = await summarize_messages(
        config.provider, old_messages, config.summary_max_tokens
    )

    # Create summary as the first message
    summary_message = Message(
        role="user",
        content=f"[Previous conversation summary]\n{summary}",
    )

    result = [summary_message, *recent_messages]

    return CompressResult(
        messages=result,
        compressed=True,
        original_count=len(messages),
        compressed_count=len(result),
        summary_tokens=estimate_message_tokens(summary_message),
    )


def needs_compression(messages: list[Message], max_tokens: int) -> bool:
    """Check if a conversation needs compression based on token count."""
    total = sum(estimate_message_tokens(m) for m in messages)
    return total > max_tokens
