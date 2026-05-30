from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Awaitable


@dataclass
class Command:
    """Built-in command handler."""

    name: str
    description: str
    execute: Callable[[], str | None]


@dataclass
class ReplConfig:
    """REPL configuration."""

    prompt: str = "> "
    exit_keywords: list[str] = field(default_factory=lambda: ["/exit", "/quit"])
    commands: list[Command] = field(default_factory=list)
    on_input: Callable[[str], Awaitable[str]] | None = None


def _default_commands() -> list[Command]:
    """Create default built-in commands."""
    return [
        Command(
            name="/help",
            description="Show available commands",
            execute=lambda: "help_placeholder",  # replaced at runtime
        ),
        Command(
            name="/clear",
            description="Clear the screen",
            execute=lambda: (sys.stdout.write("\033[2J\033[H"), None)[-1],
        ),
    ]


def format_help(commands: list[Command], exit_keywords: list[str]) -> str:
    """Format the help text listing all available commands."""
    lines = ["Available commands:"]
    for cmd in commands:
        lines.append(f"  {cmd.name:<12} {cmd.description}")
    lines.append(f"  {exit_keywords[0]:<12} Exit the REPL")
    return "\n".join(lines)


def is_multi_line(text: str) -> bool:
    """Check if the input is a multi-line paste (contains newlines)."""
    return "\n" in text


def normalize_input(text: str) -> str:
    """Normalize user input: trim whitespace."""
    return text.strip()


def parse_command(text: str) -> str:
    """Parse the command name from user input (e.g., '/help arg' -> '/help')."""
    return text.strip().split()[0].lower() if text.strip() else ""


class Repl:
    """
    Interactive REPL with built-in commands.

    For testing, use process_input() directly. For interactive use,
    call run() which reads from stdin.
    """

    def __init__(self, config: ReplConfig | None = None) -> None:
        self._config = config or ReplConfig()
        self._all_commands = _default_commands() + list(self._config.commands)

        # Fix help command to list all commands
        for cmd in self._all_commands:
            if cmd.name == "/help":
                cmd.execute = lambda: format_help(
                    self._all_commands, self._config.exit_keywords
                )
                break

    async def process_input(self, raw: str) -> str | None:
        """
        Process a single input line. Returns:
        - None if the REPL should exit
        - string response otherwise
        """
        text = normalize_input(raw)

        if not text:
            return ""

        # Check exit
        cmd_name = parse_command(text)
        if cmd_name in self._config.exit_keywords:
            return None  # signal exit

        # Check built-in commands
        for cmd in self._all_commands:
            if cmd.name == cmd_name:
                result = cmd.execute()
                return result if result is not None else ""

        # Delegate to user handler
        if self._config.on_input:
            return await self._config.on_input(text)

        return f"Unknown command: {cmd_name}. Type /help for available commands."

    async def run(self) -> None:
        """Run the REPL interactively (reads from stdin)."""
        print("AI Coding Agent (type /help for commands, /exit to quit)\n")

        while True:
            try:
                raw = input(self._config.prompt)
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            result = await self.process_input(raw)

            if result is None:
                print("Goodbye!")
                break

            if result:
                print(result)