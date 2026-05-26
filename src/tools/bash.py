from __future__ import annotations
import asyncio
from dataclasses import dataclass

from src.llm.types import Tool

# Tool definition for LLM
BASH_TOOL_DEFINITION = Tool(
    name="bash",
    description=(
        "Execute a bash command and return its output. "
        "Use this to run shell commands, scripts, or system utilities."
    ),
    input_schema={
        "type":"object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default: 30)",
            },
        },
        "required": ["command"],
    }
)

@dataclass
class BashToolInput:
    """Input parameters for the bash tool."""
    command: str
    timeout: float = 30.0

MAX_OUTPUT_SIZE = 100_000 # 100KB

def _truncate_output(output: str) -> str:
    """Truncate output if it exceeds the max size."""
    if len(output) <= MAX_OUTPUT_SIZE:
        return output
    half = MAX_OUTPUT_SIZE // 2
    return output[:half] + "\n\n... (truncated) ...\n\n" + output[-half:]


async def execute_bash_tool(input: BashToolInput) -> str:
    """Execute a bash command and return its output."""
    command = input.command
    timeout = input.timeout

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await  process.wait()
            return f"Error: command timed out after {timeout}s"

    except OSError as e:
        return f"Error: failed to execute command: {e}"

    stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

    parts: list[str] = []
    if stdout:
        parts.append(_truncate_output(stdout))
    if stderr:
        parts.append(f"STDERR:\n{_truncate_output(stderr)}")

    if process.returncode != 0:
        parts.append(f"\nExit code: {process.returncode}")

    return "\n".join(parts) or "(no output)"