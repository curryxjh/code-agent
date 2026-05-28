
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.agent import AgentConfig, AgentResult, ToolCallRecord, run_agent
from src.llm.types import (
    ChatOptions,
    ChatResponse,
    ContentBlock,
    Message,
    TextBlock,
    Tool,
    ToolUseBlock,
)


def make_provider(responses: list[ChatResponse]) -> AsyncMock:
    provider = AsyncMock()
    provider.chat = AsyncMock(side_effect=responses)
    return provider


TEST_TOOL = Tool(
    name="test_tool",
    description="A test tool",
    input_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)


def base_config(provider, execute_tool=None) -> AgentConfig:
    async def default_exec(name: str, inp: dict) -> str:
        return "tool result"

    return AgentConfig(
        provider=provider,
        system="You are a test assistant.",
        tools=[TEST_TOOL],
        execute_tool=execute_tool or default_exec,
    )


class TestRunAgent:
    @pytest.mark.asyncio
    async def test_return_text_without_tool_use(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[TextBlock(type="text", text="Hello!")],
                    text="Hello!",
                    stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 5},
                )
            ]
        )

        result = await run_agent(base_config(provider), "Hi")

        assert result.text == "Hello!"
        assert result.tool_calls == []
        assert result.iterations == 1

    @pytest.mark.asyncio
    async def test_execute_tool_calls_and_continue(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[
                        TextBlock(type="text", text="Let me check."),
                        ToolUseBlock(
                            type="tool_use",
                            id="call_1",
                            name="test_tool",
                            input={"query": "hello"},
                        ),
                    ],
                    text="Let me check.",
                    stop_reason="tool_use",
                    usage={"input_tokens": 15, "output_tokens": 10},
                ),
                ChatResponse(
                    content=[TextBlock(type="text", text="The result is ready.")],
                    text="The result is ready.",
                    stop_reason="end_turn",
                    usage={"input_tokens": 30, "output_tokens": 8},
                ),
            ]
        )

        calls = []

        async def exec_tool(name: str, inp: dict) -> str:
            calls.append((name, inp))
            return "tool output"

        result = await run_agent(base_config(provider, exec_tool), "Check something")

        assert result.text == "The result is ready."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "test_tool"
        assert result.tool_calls[0].input == {"query": "hello"}
        assert result.tool_calls[0].result == "tool output"
        assert result.iterations == 2
        assert calls == [("test_tool", {"query": "hello"})]

    @pytest.mark.asyncio
    async def test_handle_multiple_tool_calls(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[
                        ToolUseBlock(
                            type="tool_use",
                            id="call_1",
                            name="test_tool",
                            input={"query": "a"},
                        ),
                        ToolUseBlock(
                            type="tool_use",
                            id="call_2",
                            name="test_tool",
                            input={"query": "b"},
                        ),
                    ],
                    text="",
                    stop_reason="tool_use",
                    usage={"input_tokens": 10, "output_tokens": 10},
                ),
                ChatResponse(
                    content=[TextBlock(type="text", text="Done.")],
                    text="Done.",
                    stop_reason="end_turn",
                    usage={"input_tokens": 20, "output_tokens": 5},
                ),
            ]
        )

        async def exec_tool(name: str, inp: dict) -> str:
            return f"result-{inp['query']}"

        result = await run_agent(
            base_config(provider, exec_tool), "Do two things"
        )

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].result == "result-a"
        assert result.tool_calls[1].result == "result-b"

    @pytest.mark.asyncio
    async def test_stop_at_max_iterations(self):
        infinite_response = ChatResponse(
            content=[
                ToolUseBlock(
                    type="tool_use",
                    id="call_inf",
                    name="test_tool",
                    input={"query": "loop"},
                )
            ],
            text="",
            stop_reason="tool_use",
            usage={"input_tokens": 5, "output_tokens": 5},
        )

        provider = make_provider([infinite_response] * 3)

        config = base_config(provider)
        config.max_iterations = 3
        result = await run_agent(config, "Loop forever")

        assert result.text == "(max iterations reached)"
        assert result.iterations == 3
        assert len(result.tool_calls) == 3

    @pytest.mark.asyncio
    async def test_accumulate_token_usage(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[
                        ToolUseBlock(
                            type="tool_use",
                            id="c1",
                            name="test_tool",
                            input={"query": "x"},
                        )
                    ],
                    text="",
                    stop_reason="tool_use",
                    usage={"input_tokens": 100, "output_tokens": 50},
                ),
                ChatResponse(
                    content=[TextBlock(type="text", text="Final.")],
                    text="Final.",
                    stop_reason="end_turn",
                    usage={"input_tokens": 200, "output_tokens": 30},
                ),
            ]
        )

        result = await run_agent(base_config(provider), "Count tokens")

        assert result.input_tokens == 300
        assert result.output_tokens == 80

    @pytest.mark.asyncio
    async def test_pass_system_and_tools_to_provider(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[TextBlock(type="text", text="OK")],
                    text="OK",
                    stop_reason="end_turn",
                    usage={"input_tokens": 5, "output_tokens": 3},
                )
            ]
        )

        config = base_config(provider)
        config.system = "Be helpful."
        config.max_tokens = 500
        await run_agent(config, "Hi")

        provider.chat.assert_called_once()
        call_args = provider.chat.call_args
        assert call_args[0][0] == [Message(role="user", content="Hi")]
        opts = call_args[1]["options"] if "options" in call_args[1] else call_args[0][1]
        assert opts.system == "Be helpful."
        assert opts.tools == [TEST_TOOL]
        assert opts.max_tokens == 500

    @pytest.mark.asyncio
    async def test_build_correct_message_history(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[
                        TextBlock(type="text", text="Checking..."),
                        ToolUseBlock(
                            type="tool_use",
                            id="c1",
                            name="test_tool",
                            input={"query": "test"},
                        ),
                    ],
                    text="Checking...",
                    stop_reason="tool_use",
                    usage={"input_tokens": 10, "output_tokens": 10},
                ),
                ChatResponse(
                    content=[TextBlock(type="text", text="Done.")],
                    text="Done.",
                    stop_reason="end_turn",
                    usage={"input_tokens": 20, "output_tokens": 5},
                ),
            ]
        )

        await run_agent(base_config(provider), "Do it")

        # Second call should have full history
        second_call = provider.chat.call_args_list[1]
        messages = second_call[0][0]
        assert len(messages) == 3
        assert messages[0] == Message(role="user", content="Do it")
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"
        assert messages[2].content[0].type == "tool_result"

    @pytest.mark.asyncio
    async def test_empty_response(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[],
                    text="",
                    stop_reason="end_turn",
                    usage={"input_tokens": 5, "output_tokens": 0},
                )
            ]
        )

        result = await run_agent(base_config(provider), "Empty")
        assert result.text == ""
        assert result.iterations == 1