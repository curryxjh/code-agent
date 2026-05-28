from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Awaitable

from src.llm.types import (
    LLMProvider,
    Message,
    Tool,
    ChatOptions,
    ContentBlock,
)
from src.llm.helpers import extract_text, extract_tool_uses, create_tool_result

# Function that executes a tool by name and returns the result string
ToolExecutor = Callable[[str, dict], Awaitable[str]]

DEFAULT_MAX_ITERATIONS = 10


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""

    provider: LLMProvider
    system: str
    tools: list[Tool]
    execute_tool: ToolExecutor
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_tokens: int | None = None


@dataclass
class ToolCallRecord:
    """Record of a single tool call during the agent loop."""

    name: str
    input: dict
    result: str


@dataclass
class AgentResult:
    """Result returned after the agent loop completes."""

    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


async def run_agent(config: AgentConfig, user_message: str) -> AgentResult:
    """
    Run the agent loop: send user message to LLM, execute tool calls,
    feed results back, repeat until LLM stops calling tools.
    """
    messages: list[Message] = [Message(role="user", content=user_message)]
    tool_calls: list[ToolCallRecord] = []
    total_input = 0
    total_output = 0

    for i in range(config.max_iterations):
        response = await config.provider.chat(
            messages,
            options=ChatOptions(
                system=config.system,
                tools=config.tools,
                max_tokens=config.max_tokens,
            ),
        )

        total_input += response.usage.get("input_tokens", 0)
        total_output += response.usage.get("output_tokens", 0)

        # If LLM did not request tool use, we are done
        if response.stop_reason != "tool_use":
            return AgentResult(
                text=extract_text(response.content),
                tool_calls=tool_calls,
                iterations=i + 1,
                input_tokens=total_input,
                output_tokens=total_output,
            )

        # Extract tool use blocks and add assistant message to history
        uses = extract_tool_uses(response.content)
        messages.append(Message(role="assistant", content=response.content))

        # Execute each tool call sequentially
        results: list[ContentBlock] = []
        for use in uses:
            result = await config.execute_tool(use.name, use.input)
            tool_calls.append(
                ToolCallRecord(name=use.name, input=use.input, result=result)
            )
            results.append(create_tool_result(use.id, result))

        # Add tool results as user message
        messages.append(Message(role="user", content=results))

    # Max iterations reached without LLM finishing
    return AgentResult(
        text="(max iterations reached)",
        tool_calls=tool_calls,
        iterations=config.max_iterations,
        input_tokens=total_input,
        output_tokens=total_output,
    )