from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.anthropic_provider import AnthropicProvider, AnthropicConfig
from src.llm.openai_compatible import OpenAICompatibleProvider, OpenAICompatibleConfig
from src.llm.types import Message, StreamEvent


async def collect_events(stream) -> list[StreamEvent]:
    """Collect all events from an async iterator."""
    events = []
    async for event in stream:
        events.append(event)
    return events


@pytest.mark.asyncio
class TestAnthropicStreaming:
    async def test_yield_stream_events(self):
        with patch("src.llm.anthropic_provider.AsyncAnthropic"):
            provider = AnthropicProvider(AnthropicConfig(api_key="test-key"))

        # Mock the stream context manager
        mock_event1 = MagicMock()
        mock_event1.type = "content_block_delta"
        mock_event1.delta.type = "text_delta"
        mock_event1.delta.text = "Hello"

        mock_event2 = MagicMock()
        mock_event2.type = "content_block_delta"
        mock_event2.delta.type = "text_delta"
        mock_event2.delta.text = " world"

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        async def async_iter():
            yield mock_event1
            yield mock_event2

        mock_stream.__aiter__ = lambda self: async_iter()

        provider._client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [Message(role="user", content="Hi")]
        events = await collect_events(provider.stream(messages))

        assert events[0].type == "message_start"
        assert events[1] == StreamEvent(type="text_delta", text="Hello")
        assert events[2] == StreamEvent(type="text_delta", text=" world")
        assert events[-1].type == "message_stop"

    async def test_full_text_from_deltas(self):
        with patch("src.llm.anthropic_provider.AsyncAnthropic"):
            provider = AnthropicProvider(AnthropicConfig(api_key="test-key"))

        mock_event1 = MagicMock()
        mock_event1.type = "content_block_delta"
        mock_event1.delta.type = "text_delta"
        mock_event1.delta.text = "Hello"

        mock_event2 = MagicMock()
        mock_event2.type = "content_block_delta"
        mock_event2.delta.type = "text_delta"
        mock_event2.delta.text = " world"

        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
        mock_stream.__aexit__ = AsyncMock(return_value=False)
        async def async_iter():
            yield mock_event1
            yield mock_event2

        mock_stream.__aiter__ = lambda self: async_iter()

        provider._client.messages.stream = MagicMock(return_value=mock_stream)

        messages = [Message(role="user", content="Hi")]
        full_text = ""
        async for event in provider.stream(messages):
            if event.type == "text_delta" and event.text:
                full_text += event.text
        assert full_text == "Hello world"


@pytest.mark.asyncio
class TestOpenAICompatibleStreaming:
    async def test_yield_stream_events(self):
        with patch("src.llm.openai_compatible.AsyncOpenAI"):
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="test-key",
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat",
                )
            )

        # Mock chunks as async iterable
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"

        async def mock_create(**kwargs):
            async def gen():
                yield mock_chunk1
                yield mock_chunk2
            return gen()

        provider._client.chat.completions.create = mock_create

        messages = [Message(role="user", content="Hi")]
        events = await collect_events(provider.stream(messages))

        assert events[0].type == "message_start"
        assert events[1] == StreamEvent(type="text_delta", text="Hello")
        assert events[2] == StreamEvent(type="text_delta", text=" world")
        assert events[-1].type == "message_stop"

    async def test_full_text_from_deltas(self):
        with patch("src.llm.openai_compatible.AsyncOpenAI"):
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="test-key",
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat",
                )
            )

        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = " world"

        async def mock_create(**kwargs):
            async def gen():
                yield mock_chunk1
                yield mock_chunk2
            return gen()

        provider._client.chat.completions.create = mock_create

        messages = [Message(role="user", content="Hi")]
        full_text = ""
        async for event in provider.stream(messages):
            if event.type == "text_delta" and event.text:
                full_text += event.text
        assert full_text == "Hello world"