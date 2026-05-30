from __future__ import annotations

import os
import sys
import unicodedata
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


def _char_width(ch: str) -> int:
    """Terminal display width of a single character."""
    cp = ord(ch)
    if cp > 0xFFFF:
        return 2
    if unicodedata.east_asian_width(ch) in ("F", "W"):
        return 2
    return 1


def _read_exact(fd: int, n: int) -> bytes:
    """Read exactly *n* bytes from *fd* (blocks until all arrive)."""
    buf = b""
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if not chunk:
            return buf
        buf += chunk
    return buf


def _read_line(prompt: str) -> str | None:
    """
    Read a line using cbreak mode with manual UTF-8 decoding.

    Bypasses readline / libedit entirely so that CJK backspace works
    correctly regardless of terminal IUTF8 settings.
    Returns None on EOF (Ctrl-D) or Ctrl-C.
    """
    import tty
    import termios

    sys.stdout.write(prompt)
    sys.stdout.flush()

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        chars: list[str] = []

        while True:
            b = os.read(fd, 1)
            if not b:
                return None

            byte = b[0]

            # Enter
            if byte in (0x0A, 0x0D):
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return "".join(chars)

            # Backspace / DEL
            if byte in (0x7F, 0x08):
                if chars:
                    removed = chars.pop()
                    w = _char_width(removed)
                    sys.stdout.write("\b \b" * w)
                    sys.stdout.flush()
                continue

            # Ctrl-C
            if byte == 0x03:
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None

            # Ctrl-D — EOF when line is empty
            if byte == 0x04:
                if not chars:
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    return None
                continue

            # Ctrl-U — clear entire line
            if byte == 0x15:
                while chars:
                    w = _char_width(chars.pop())
                    sys.stdout.write("\b \b" * w)
                sys.stdout.flush()
                continue

            # Escape sequences (arrow keys, etc.) — skip
            if byte == 0x1B:
                next_b = os.read(fd, 1)
                if next_b and next_b[0] == 0x5B:
                    os.read(fd, 1)
                continue

            # Skip other control characters
            if byte < 0x20:
                continue

            # Decode UTF-8 (1–4 bytes).
            # Use _read_exact to guarantee all continuation bytes arrive
            # (cbreak VMIN=1 may return partial reads from os.read).
            if byte < 0x80:
                char = chr(byte)
            elif byte < 0xC0:
                continue  # stray continuation byte
            elif byte < 0xE0:
                char = (b + _read_exact(fd, 1)).decode("utf-8", errors="replace")
            elif byte < 0xF0:
                char = (b + _read_exact(fd, 2)).decode("utf-8", errors="replace")
            else:
                char = (b + _read_exact(fd, 3)).decode("utf-8", errors="replace")

            if char and char != "\ufffd":
                chars.append(char)
                sys.stdout.write(char)
                sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


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
        prompt = self._config.prompt
        use_cbreak = sys.stdin.isatty()

        while True:
            try:
                if use_cbreak:
                    raw = _read_line(prompt)
                    if raw is None:
                        print("Goodbye!")
                        break
                else:
                    # Non-TTY: fall back to simple line reading
                    sys.stdout.write(prompt)
                    sys.stdout.flush()
                    raw = sys.stdin.readline()
                    if not raw:
                        break
                    raw = raw.rstrip("\n")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            result = await self.process_input(raw)

            if result is None:
                print("Goodbye!")
                break

            if result:
                print(result)