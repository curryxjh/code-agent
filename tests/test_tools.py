import pytest

from src.llm.types import Tool, TextBlock, ToolUseBlock, ToolResultBlock, Message
from src.llm.anthropic_provider import AnthropicProvider, AnthropicConfig
from src.llm.openai_compatible import OpenAICompatibleProvider, OpenAICompatibleConfig
from unittest.mock import patch, AsyncMock, MagicMock
from src.llm.helpers import create_tool_result, extract_text, extract_tool_uses


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
        assert response.content[1] == ToolUseBlock(
            id="call_123", name="read_file", input={"path": "tests/test.txt"}
        )

        # Verify tools were sent
        call_kwargs = provider._client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"][0]["name"] == "read_file"


    async def test_content_block_messages(self):
        with patch("src.llm.anthropic_provider.AsyncAnthropic"):
            provider = AnthropicProvider(AnthropicConfig(api_key="test-key"))

        mock_text = MagicMock()
        mock_text.type = "text"
        mock_text.text = "Done."
        mock_response = MagicMock()
        mock_response.content = [mock_text]
        mock_response.stop_reason = "end_turn"
        mock_response.usage.input_tokens = 30
        mock_response.usage.output_tokens = 5

        provider._client.messages.create = AsyncMock(return_value=mock_response)

        messages = [
            Message(role="user", content="Read main.ts"),
            Message(role="assistant", content=[
                TextBlock(text="Reading..."),
                ToolUseBlock(id="call_123", name="read_file", input={"path": "tests/test.txt"}),
            ]),
            Message(role="user", content=[
                ToolResultBlock(tool_use_id="call_123", content="console.log('this is a test file.');"),
            ]),
        ]

        await provider.chat(messages)

        call_kwargs = provider._client.messages.create.call_args[1]
        assistant_msg = call_kwargs["messages"][1]
        assert assistant_msg["content"][0] == {"type": "text", "text": "Reading..."}
        assert assistant_msg["content"][1]["type"] == "tool_use"

        user_msg = call_kwargs["messages"][2]
        assert user_msg["content"][0]["type"] == "tool_result"
        assert user_msg["content"][0]["tool_use_id"] == "call_123"

    @pytest.mark.asyncio
    class TestOpenAICompatibleProviderWithTools:
        async def test_tool_use_response(self):
            with patch("src.llm.openai_compatible.AsyncOpenAI"):
                provider = OpenAICompatibleProvider(
                    OpenAICompatibleConfig(
                        api_key="test-key",
                        base_url="https://api.deepseek.com",
                        model="deepseek-chat",
                    )
                )

            mock_tc = MagicMock()
            mock_tc.id = "call_456"
            mock_tc.type = "function"
            mock_tc.function.name = "read_file"
            mock_tc.function.arguments = '{"path":"/src/main.ts"}'

            mock_choice = MagicMock()
            mock_choice.message.content = "Let me read that."
            mock_choice.message.tool_calls = [mock_tc]
            mock_choice.finish_reason = "tool_calls"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage.prompt_tokens = 20
            mock_response.usage.completion_tokens = 15

            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            from src.llm.types import ChatOptions
            messages = [Message(role="user", content="Read main.ts")]
            response = await provider.chat(messages, ChatOptions(tools=[TEST_TOOL]))

            assert response.stop_reason == "tool_use"
            assert len(response.content) == 2
            assert response.content[0] == TextBlock(text="Let me read that.")
            assert response.content[1] == ToolUseBlock(
                id="call_456", name="read_file", input={"path": "/src/main.ts"}
            )

        async def test_content_block_messages(self):
            with patch("src.llm.openai_compatible.AsyncOpenAI"):
                provider = OpenAICompatibleProvider(
                    OpenAICompatibleConfig(
                        api_key="test-key",
                        base_url="https://api.deepseek.com",
                        model="deepseek-chat",
                    )
                )

            mock_choice = MagicMock()
            mock_choice.message.content = "The file contains hello."
            mock_choice.message.tool_calls = None
            mock_choice.finish_reason = "stop"

            mock_response = MagicMock()
            mock_response.choices = [mock_choice]
            mock_response.usage.prompt_tokens = 30
            mock_response.usage.completion_tokens = 5

            provider._client.chat.completions.create = AsyncMock(
                return_value=mock_response
            )

            messages = [
                Message(role="user", content="Read main.ts"),
                Message(role="assistant", content=[
                    TextBlock(text="Reading..."),
                    ToolUseBlock(id="call_456", name="read_file", input={"path": "/src/main.ts"}),
                ]),
                Message(role="user", content=[
                    ToolResultBlock(tool_use_id="call_456", content="console.log('hello');"),
                ]),
            ]

            await provider.chat(messages)

            call_kwargs = provider._client.chat.completions.create.call_args[1]
            assistant_msg = call_kwargs["messages"][1]
            assert "tool_calls" in assistant_msg
            assert assistant_msg["tool_calls"][0]["function"]["name"] == "read_file"

            tool_msg = call_kwargs["messages"][2]
            assert tool_msg["role"] == "tool"
            assert tool_msg["tool_call_id"] == "call_456"

    class TestHelperFunctions:
        def test_extract_text_joins_text_blocks(self):
            content = [
                TextBlock(text="Hello "),
                ToolUseBlock(id="1", name="test", input={}),
                TextBlock(text="world"),
            ]
            assert extract_text(content) == "Hello world"

        def test_extract_text_empty_for_no_text_blocks(self):
            content = [ToolUseBlock(id="1", name="test", input={})]
            assert extract_text(content) == ""

        def test_extract_tool_uses(self):
            content = [
                TextBlock(text="Hello"),
                ToolUseBlock(id="1", name="read_file", input={"path": "a.ts"}),
                ToolUseBlock(id="2", name="write_file", input={"path": "b.ts"}),
            ]
            tools = extract_tool_uses(content)
            assert len(tools) == 2
            assert tools[0].name == "read_file"
            assert tools[1].name == "write_file"

        def test_create_tool_result(self):
            result = create_tool_result("call_1", "file contents")
            assert result == ToolResultBlock(
                tool_use_id="call_1", content="file contents"
            )

        def test_create_tool_result_with_error(self):
            result = create_tool_result("call_1", "not found", is_error=True)
            assert result == ToolResultBlock(
                tool_use_id="call_1", content="not found", is_error=True
            )