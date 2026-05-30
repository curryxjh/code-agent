from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.compressor import (
    CompressorConfig,
    summarize_messages,
    compress_conversation,
    needs_compression,
)
from src.llm.types import (
    Message,
    ChatResponse,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
)


def _mock_provider(summary_text: str = "Summary of the conversation."):
    provider = AsyncMock()
    provider.chat = AsyncMock(
        return_value=ChatResponse(
            content=[TextBlock(text=summary_text)],
            text=summary_text,
            stop_reason="end_turn",
            usage={"input_tokens": 100, "output_tokens": 20},
        )
    )
    return provider


def _user_msg(text: str) -> Message:
    return Message(role="user", content=text)


def _assistant_msg(text: str) -> Message:
    return Message(role="assistant", content=text)


def _build_conversation(pairs: int) -> list[Message]:
    msgs: list[Message] = []
    for i in range(pairs):
        msgs.append(_user_msg(f"Question {i}: {'x' * 200}"))
        msgs.append(_assistant_msg(f"Answer {i}: {'y' * 200}"))
    return msgs


class TestSummarizeMessages:
    @pytest.mark.asyncio
    async def test_returns_summary(self):
        provider = _mock_provider("This is the summary.")
        msgs = [_user_msg("Hello"), _assistant_msg("Hi there!")]
        summary = await summarize_messages(provider, msgs)
        assert summary == "This is the summary."
        assert provider.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_formats_tool_blocks(self):
        provider = _mock_provider("Summary with tools.")
        msgs = [
            Message(
                role="assistant",
                content=[ToolUseBlock(id="1", name="read_file", input={"file_path": "test.txt"})],
            ),
            Message(
                role="user",
                content=[ToolResultBlock(tool_use_id="1", content="file contents...")],
            ),
        ]
        summary = await summarize_messages(provider, msgs)
        assert summary == "Summary with tools."

        # Verify formatted input contains tool info
        call_args = provider.chat.call_args
        input_content = call_args[0][0][0].content
        assert "[Tool call: read_file]" in input_content
        assert "[Tool result:" in input_content


class TestCompressConversation:
    @pytest.mark.asyncio
    async def test_no_compress_under_budget(self):
        config = CompressorConfig(
            provider=_mock_provider(),
            max_tokens=100000,
            keep_recent_messages=4,
        )
        msgs = [_user_msg("Hi"), _assistant_msg("Hello")]
        result = await compress_conversation(config, msgs)

        assert result.compressed is False
        assert len(result.messages) == 2
        assert result.original_count == 2
        assert result.compressed_count == 2
        assert result.summary_tokens == 0

    @pytest.mark.asyncio
    async def test_no_compress_few_messages(self):
        config = CompressorConfig(
            provider=_mock_provider(),
            max_tokens=10,
            keep_recent_messages=4,
        )
        msgs = [_user_msg("A"), _assistant_msg("B")]
        result = await compress_conversation(config, msgs)
        assert result.compressed is False

    @pytest.mark.asyncio
    async def test_compress_over_budget(self):
        provider = _mock_provider("Compressed summary.")
        config = CompressorConfig(
            provider=provider,
            max_tokens=100,
            keep_recent_messages=2,
        )
        msgs = _build_conversation(10)  # 20 messages
        result = await compress_conversation(config, msgs)

        assert result.compressed is True
        assert result.original_count == 20
        assert result.compressed_count == 3  # summary + 2 recent
        assert result.summary_tokens > 0

        first_content = result.messages[0].content
        assert isinstance(first_content, str)
        assert "[Previous conversation summary]" in first_content
        assert "Compressed summary." in first_content

        assert result.messages[-1] is msgs[-1]

    @pytest.mark.asyncio
    async def test_calls_provider(self):
        provider = _mock_provider("sum")
        config = CompressorConfig(
            provider=provider,
            max_tokens=10,
            keep_recent_messages=2,
        )
        msgs = _build_conversation(5)
        await compress_conversation(config, msgs)
        assert provider.chat.call_count == 1


class TestNeedsCompression:
    def test_short_conversation(self):
        msgs = [_user_msg("Hi"), _assistant_msg("Hello")]
        assert needs_compression(msgs, 100000) is False

    def test_long_conversation(self):
        msgs = _build_conversation(50)
        assert needs_compression(msgs, 100) is True

    def test_empty(self):
        assert needs_compression([], 1000) is False