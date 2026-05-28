from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.agent import AgentConfig, run_agent
from src.llm.types import (
    ChatResponse,
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


class TestParallelToolCalls:
    @pytest.mark.asyncio
    async def test_execute_concurrently_when_enabled(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[
                        ToolUseBlock(type="tool_use", id="c1", name="test_tool", input={"query": "a"}),
                        ToolUseBlock(type="tool_use", id="c2", name="test_tool", input={"query": "b"}),
                        ToolUseBlock(type="tool_use", id="c3", name="test_tool", input={"query": "c"}),
                    ],
                    text="",
                    stop_reason="tool_use",
                    usage={"input_tokens": 10, "output_tokens": 10},
                ),
                ChatResponse(
                    content=[TextBlock(type="text", text="All done.")],
                    text="All done.",
                    stop_reason="end_turn",
                    usage={"input_tokens": 20, "output_tokens": 5},
                ),
            ]
        )

        start_times: list[float] = []

        async def exec_tool(name: str, inp: dict) -> str:
            start_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return f"result-{inp['query']}"

        result = await run_agent(
            AgentConfig(
                provider=provider,
                system="test",
                tools=[TEST_TOOL],
                execute_tool=exec_tool,
                parallel_tool_calls=True,
            ),
            "Do three things",
        )

        assert len(result.tool_calls) == 3
        assert result.tool_calls[0].result == "result-a"
        assert result.tool_calls[1].result == "result-b"
        assert result.tool_calls[2].result == "result-c"

        # All three should start within a small window (parallel)
        span = max(start_times) - min(start_times)
        assert span < 0.03

    @pytest.mark.asyncio
    async def test_execute_sequentially_when_disabled(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[
                        ToolUseBlock(type="tool_use", id="c1", name="test_tool", input={"query": "a"}),
                        ToolUseBlock(type="tool_use", id="c2", name="test_tool", input={"query": "b"}),
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

        start_times: list[float] = []

        async def exec_tool(name: str, inp: dict) -> str:
            start_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return f"result-{inp['query']}"

        await run_agent(
            AgentConfig(
                provider=provider,
                system="test",
                tools=[TEST_TOOL],
                execute_tool=exec_tool,
                parallel_tool_calls=False,
            ),
            "Do two things",
        )

        # Sequential: second should start after first finishes (~50ms gap)
        assert start_times[1] - start_times[0] >= 0.04

    @pytest.mark.asyncio
    async def test_preserve_result_order(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[
                        ToolUseBlock(type="tool_use", id="c1", name="test_tool", input={"query": "slow"}),
                        ToolUseBlock(type="tool_use", id="c2", name="test_tool", input={"query": "fast"}),
                    ],
                    text="",
                    stop_reason="tool_use",
                    usage={"input_tokens": 10, "output_tokens": 10},
                ),
                ChatResponse(
                    content=[TextBlock(type="text", text="OK")],
                    text="OK",
                    stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 5},
                ),
            ]
        )

        async def exec_tool(name: str, inp: dict) -> str:
            delay = 0.08 if inp["query"] == "slow" else 0.01
            await asyncio.sleep(delay)
            return f"done-{inp['query']}"

        result = await run_agent(
            AgentConfig(
                provider=provider,
                system="test",
                tools=[TEST_TOOL],
                execute_tool=exec_tool,
                parallel_tool_calls=True,
            ),
            "Order test",
        )

        # Results should match tool call order, not completion order
        assert result.tool_calls[0].result == "done-slow"
        assert result.tool_calls[1].result == "done-fast"

    @pytest.mark.asyncio
    async def test_single_tool_call_with_parallel_enabled(self):
        provider = make_provider(
            [
                ChatResponse(
                    content=[
                        ToolUseBlock(type="tool_use", id="c1", name="test_tool", input={"query": "only"}),
                    ],
                    text="",
                    stop_reason="tool_use",
                    usage={"input_tokens": 10, "output_tokens": 10},
                ),
                ChatResponse(
                    content=[TextBlock(type="text", text="OK")],
                    text="OK",
                    stop_reason="end_turn",
                    usage={"input_tokens": 10, "output_tokens": 5},
                ),
            ]
        )

        call_count = 0

        async def exec_tool(name: str, inp: dict) -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        result = await run_agent(
            AgentConfig(
                provider=provider,
                system="test",
                tools=[TEST_TOOL],
                execute_tool=exec_tool,
                parallel_tool_calls=True,
            ),
            "Single tool",
        )

        assert len(result.tool_calls) == 1
        assert call_count == 1