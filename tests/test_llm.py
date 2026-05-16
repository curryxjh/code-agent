import logging
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from src.llm.anthropic_provider import AnthropicConfig, AnthropicProvider
from src.llm.factory import ProviderConfig, create_provider
from src.llm.openai_compatible import OpenAICompatibleConfig, OpenAICompatibleProvider
from src.llm.types import ChatOptions, Message



Logger = logging.getLogger(__name__)

# ── AnthropicProvider ──

@pytest.mark.asyncio
class TestAnthropicProvider:
    async def test_send_message_and_return_response(self):
        with patch("src.llm.anthropic_provider.AsyncAnthropic"):
            provider = AnthropicProvider(AnthropicConfig(api_key="test-key"))

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hello from Claude!"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        provider._client.messages.create = AsyncMock(return_value=mock_response)

        message = [Message(role="user", content="Hi")]
        response = await provider.chat(message)

        assert response.text == "Hello from Claude!"
        assert response.stop_reason == "end_turn"
        assert response.usage["input_tokens"] == 10
        assert response.usage["output_tokens"] == 5

    async def test_pass_system_prompt_and_max_tokens(self):
        with patch("src.llm.anthropic_provider.AsyncAnthropic"):
            provider = AnthropicProvider(
                AnthropicConfig(api_key="tests-key", model="claude-haiku-4-5-20251001"),
            )

        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = "Hello from Claude!"
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 1

        provider._client.messages.create = AsyncMock(return_value=mock_response)

        messages = [Message(role="user", content="Hi")]
        await provider.chat(messages, ChatOptions(system="Be helpful.", max_tokens=1024))

        call_kwargs = provider._client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be helpful."
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"


# ── OpenAICompatibleProvider ──

@pytest.mark.asyncio
class TestOpenAICompatibleProvider:
    async def test_send_message_and_return_response(self):
        with patch("src.llm.openai_compatible.AsyncOpenAI"):
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="tests-key",
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat",
                )
            )

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from DeepSeek!"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        messages = [Message(role="user", content="Hi")]
        response = await provider.chat(messages)

        assert response.text == "Hello from DeepSeek!"
        assert response.stop_reason == "end_turn"
        assert response.usage["input_tokens"] == 10
        assert response.usage["output_tokens"] == 5

    async def test_prepend_system_message(self):
        with patch("src.llm.openai_compatible.AsyncOpenAI"):
            provider = OpenAICompatibleProvider(
                OpenAICompatibleConfig(
                    api_key="tests-key",
                    base_url="https://api.deepseek.com",
                    model="deepseek-chat",
                )
            )

        mock_choice = MagicMock()
        mock_choice.message.content = "Hi"
        mock_choice.finish_reason = "stop"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 1

        provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        messages = [Message(role="user", content="Hi")]
        await provider.chat(messages, ChatOptions(system="Be helpful."))

        call_kwargs = provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0] == {
            "role": "system",
            "content": "Be helpful.",
        }

# ── Factory ──

class TestCreateProvider:
    @patch("src.llm.anthropic_provider.AsyncAnthropic")
    def test_create_anthropic_provider(self, _mock):
        p = create_provider(ProviderConfig(provider="anthropic", api_key="key"))
        print("type p: ", type(p))
        print(AnthropicProvider)
        print(type(p) is AnthropicProvider)
        print(type(p).__module__, AnthropicProvider.__module__)
        assert isinstance(p, AnthropicProvider)

    @patch("src.llm.openai_compatible.AsyncOpenAI")
    def test_create_openai_compatible_provider(self, _mock):
        p = create_provider(
            ProviderConfig(
                provider="openai-compatible",
                api_key="key",
                base_url="https://api.example.com",
                model="model-1",
            )
        )

        Logger.info(type(p))
        assert isinstance(p, OpenAICompatibleProvider)

    def test_throw_if_base_url_missing(self):
        with pytest.raises(ValueError, match="base_url"):
            create_provider(
                ProviderConfig(
                    provider="openai-compatible", api_key="key", model="m"
                )
            )

    def test_throw_if_model_missing(self):
        with pytest.raises(ValueError, match="model"):
            create_provider(
                ProviderConfig(
                    provider="openai-compatible",
                    api_key="key",
                    base_url="https://api.example.com",
                )
            )

    def test_throw_for_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_provider(ProviderConfig(provider="unknown", api_key="key"))