import pytest

from src.llm.types import Tool, TextBlock, ToolUseBlock, ToolResultBlock, Message
from src.llm.anthropic_provider import AnthropicProvider, AnthropicConfig
from src.llm.openai_compatible import OpenAICompatibleProvider, OpenAICompatibleConfig
from unittest.mock import patch, AsyncMock, MagicMock


TEST_TOOL = Tool(
    name="read_file",
    description="Reads a file",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"}
        },
        "required": ["path"]
    }
)

@pytest.mark.asyncio
class TestAnthropicProviderWithTools:
    async def test_tool_use_response(self):
        with patch("src.llm.anthropic_provider.AsyncAnthropic"):
            provider = AnthropicProvider(AnthropicConfig(api_key="test-key"))

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Let me read that file."
        mock_tool = MagicMock()
        mock_tool.type = "tool_use"
        mock_tool.id = "call_123"
        mock_tool.name = "read_file"
        mock_tool.input = {"path": "tests/test.txt"}

        mock_response = MagicMock()
        mock_response.content = [mock_text, mock_tool]
        mock_response.stop_reason = "tool_use"
        mock_response.usage.input_tokens = 20
        mock_response.usage.output_tokens = 15

        provider._client.messages.create = AsyncMock(return_value=mock_response)

        from src.llm.types import ChatOptions
        messages = [Message(role="user", content="Read the file tests/test.txt")]
        response = await provider.chat(messages, ChatOptions(tools=[TEST_TOOL]))

        assert response.stop_reason == "tool_use"
        assert len(response.content) == 2
        assert response.content[0] == TextBlock(text="Let me read that file.")