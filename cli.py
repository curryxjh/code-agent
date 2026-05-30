"""
CLI entry point — ties together all 20 chapters into a working agent.

Configurable via environment variables:
  AGENT_NAME   — display name shown in banner and prompt (default: "AI Coding")
  AGENT_ICON   — emoji icon for banner and prompt (default: "🤖")
  DEEPSEEK_API_KEY / OPENAI_API_KEY — LLM API key (required)
  LLM_BASE_URL — API base URL (default: https://api.deepseek.com)
  LLM_MODEL    — model name (default: deepseek-chat)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
import unicodedata

# Ensure `from src.xxx` works when running from any directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from src.llm.factory import ProviderConfig, create_provider
from src.llm.types import Tool
from src.agent import AgentConfig, run_agent
from src.task import TaskManager, execute_task_tool, TASK_TOOLS
from src.errors import RetryProvider, RetryConfig, safe_tool_executor
from src.system_prompt import SystemPromptBuilder
from src.context import Scratchpad, SCRATCHPAD_TOOLS, execute_scratchpad_tool
from src.markdown import render_markdown
from src.tool_display import Spinner, format_tool_cycle
from src.safety import (
    FileSystemSandbox,
    check_dangerous_command,
    read_project_config,
)
from src.repl import Repl, ReplConfig, Command
from src.tools import (
    READ_TOOL_DEFINITION, ReadToolInput, execute_read_tool,
    WRITE_TOOL_DEFINITION, WriteToolInput, execute_write_tool,
    BASH_TOOL_DEFINITION, BashToolInput, execute_bash_tool,
    GLOB_TOOL_DEFINITION, GlobToolInput, execute_glob_tool,
    GREP_TOOL_DEFINITION, GrepToolInput, execute_grep_tool,
)

# ── Banner ──────────────────────────────────────────────────
_W = 58  # box inner width between ║ and ║
_R = "\033[38;5;204m"    # rose
_RB = "\033[1;38;5;204m"  # rose bold
_G = "\033[38;5;247m"    # gray
_P = "\033[38;5;218m"    # pink (light)
_X = "\033[0m"           # reset


def _dw(s: str) -> int:
    """Terminal display width (emoji/CJK = 2 cells)."""
    w = 0
    for ch in s:
        if ord(ch) > 0xFFFF:
            w += 2
        elif unicodedata.east_asian_width(ch) in ("F", "W"):
            w += 2
        else:
            w += 1
    return w


def _row(text: str, *color_parts: str) -> str:
    """Build a box row: ║ + colored content padded to _W + ║"""
    pad = " " * (_W - _dw(text))
    inner = "".join(color_parts) + pad + _R
    return f"║{inner}║"


def _show_banner(name: str, icon: str, model: str, project_dir: str) -> None:
    """Display the startup banner."""
    cwd = project_dir
    if len(cwd) > 40:
        cwd = "..." + cwd[-37:]

    title = f"     {icon}  {name}"
    subtitle = "     Your AI Coding Assistant"

    border = "═" * _W
    blank = _row("", "")

    print(f"""
{_R}╔{border}╗
{blank}
{_row(title, f"     {icon}  ", _RB, name, _X, _R)}
{blank}
{_row(subtitle, _G, subtitle)}
{blank}
{_row(f"     Model:  {model}", _G, "     Model:  ", _P, model)}
{_row(f"     Dir:    {cwd}", _G, "     Dir:    ", _P, cwd)}
{blank}
{_row("     Type /help for commands · /exit to quit", _G, "     Type ", _P, "/help", _G, " for commands · ", _P, "/exit", _G, " to quit")}
{blank}
╚{border}╝{_X}
""")


def main() -> None:
    """Entry point for the coding agent CLI."""
    # ── Configuration ───────────────────────────────────────
    agent_name = os.environ.get("AGENT_NAME", "AI Coding")
    agent_icon = os.environ.get("AGENT_ICON", "🤖")
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    model = os.environ.get("LLM_MODEL", "deepseek-chat")
    project_dir = os.getcwd()

    if not api_key:
        print("Error: Set DEEPSEEK_API_KEY or OPENAI_API_KEY environment variable.")
        raise SystemExit(1)

    # ── LLM Provider (Ch01 + Ch12 retry) ────────────────────
    base_provider = create_provider(
        ProviderConfig(
            provider="openai-compatible",
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
    )
    provider = RetryProvider(
        base_provider,
        RetryConfig(max_retries=2, base_delay=0.5, max_delay=5.0),
    )

    # ── Tools (Ch05-08 + Ch11 + Ch15) ───────────────────────
    task_manager = TaskManager()
    scratchpad = Scratchpad()

    all_tools: list[Tool] = [
        READ_TOOL_DEFINITION, WRITE_TOOL_DEFINITION, BASH_TOOL_DEFINITION,
        GLOB_TOOL_DEFINITION, GREP_TOOL_DEFINITION,
        *TASK_TOOLS,
        *SCRATCHPAD_TOOLS,
    ]

    # ── Safety (Ch20) ───────────────────────────────────────
    sandbox = FileSystemSandbox([project_dir, tempfile.gettempdir()])

    # ── Tool Executor (Ch09 + Ch12 + Ch19 + Ch20) ──────────
    async def raw_executor(name: str, inp: dict) -> str:
        if name.startswith("task_"):
            return execute_task_tool(task_manager, name, inp)
        if name.startswith("scratchpad_"):
            return execute_scratchpad_tool(scratchpad, name, inp)

        if name in ("read_file", "write_file"):
            file_path = inp.get("file_path", "")
            blocked = sandbox.check(file_path)
            if blocked:
                return blocked

        if name == "bash":
            command = inp.get("command", "")
            danger = check_dangerous_command(command)
            if danger:
                return f"⚠️ Blocked: {danger}. This command requires user confirmation."

        spinner = Spinner(f"{name}...")
        spinner.start()
        start = time.monotonic()

        try:
            if name == "read_file":
                result = execute_read_tool(ReadToolInput(**inp))
            elif name == "write_file":
                result = execute_write_tool(WriteToolInput(**inp))
            elif name == "bash":
                result = await execute_bash_tool(BashToolInput(**inp))
            elif name == "glob":
                result = execute_glob_tool(GlobToolInput(**inp))
            elif name == "grep":
                result = execute_grep_tool(GrepToolInput(**inp))
            else:
                result = f'Error: unknown tool "{name}"'

            ms = (time.monotonic() - start) * 1000
            spinner.succeed(f"{name} [{round(ms)}ms]")
            return result
        except Exception:
            spinner.fail(f"{name} failed")
            raise

    known_tools = {t.name for t in all_tools}
    execute_tool = safe_tool_executor(raw_executor, known_tools)

    # ── System Prompt (Ch13 + Ch20) ─────────────────────────
    project_config = read_project_config(project_dir)
    prompt_builder = (
        SystemPromptBuilder()
        .set_role(
            "You are a coding assistant. Help the user with software engineering tasks "
            "by reading files, writing code, and running commands. Be concise and accurate."
        )
        .add_rules([
            "Always read a file before modifying it.",
            "Explain what you are about to do before using tools.",
            "If a task is complex, break it into steps using task tools.",
            "Never execute destructive commands without confirmation.",
            "Use the scratchpad to track your plan and findings.",
        ])
        .add_tool_guide(all_tools)
        .set_output_constraints(
            "Respond in the user's language. Use markdown for code blocks. Keep explanations brief."
        )
    )

    if project_config:
        prompt_builder.add_section("Project Instructions", project_config, 90)

    system_prompt = prompt_builder.build()

    # ── REPL (Ch17) ─────────────────────────────────────────
    prompt_str = f"{_R}{agent_icon} > {_X}"

    async def on_input(text: str) -> str:
        config = AgentConfig(
            provider=provider,
            system=system_prompt,
            tools=all_tools,
            execute_tool=execute_tool,
            max_iterations=50,
            max_tokens=4096,
            parallel_tool_calls=True,
        )
        result = await run_agent(config, text)
        return render_markdown(result.text)

    repl = Repl(
        ReplConfig(
            prompt=prompt_str,
            commands=[
                Command(
                    name="/tasks",
                    description="Show current tasks",
                    execute=lambda: task_manager.format_for_llm() or "No tasks.",
                ),
                Command(
                    name="/notes",
                    description="Show scratchpad",
                    execute=lambda: scratchpad.format() or "Scratchpad is empty.",
                ),
                Command(
                    name="/reset",
                    description="Clear tasks and scratchpad",
                    execute=lambda: (task_manager.clear(), scratchpad.clear(), "Cleared.")[-1],
                ),
            ],
            on_input=on_input,
        )
    )

    async def _run() -> None:
        _show_banner(agent_name, agent_icon, model, project_dir)
        await repl.run()

    asyncio.run(_run())


if __name__ == "__main__":
    main()