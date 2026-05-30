"""
Tool call visualization: spinner animation, parameter display,
execution timing, and output folding.
"""
from __future__ import annotations

import json
import sys
import threading
import time

from src.markdown import (
    RESET,
    BOLD,
    DIM,
    CYAN,
    GREEN,
    YELLOW,
    MAGENTA,
    GRAY,
)


class Spinner:
    """
    Spinner animation for long-running operations.

    Shows a rotating character sequence with a message, updating
    in-place using terminal control codes.
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str) -> None:
        self._message = message
        self._frame_index = 0
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def message(self) -> str:
        return self._message

    def current_frame(self) -> str:
        """Get the current frame character."""
        return self.FRAMES[self._frame_index % len(self.FRAMES)]

    def start(self) -> None:
        """Start the spinner animation."""
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        while self._running:
            frame = self.FRAMES[self._frame_index % len(self.FRAMES)]
            sys.stderr.write(f"\r{CYAN}{frame}{RESET} {self._message}")
            sys.stderr.flush()
            self._frame_index += 1
            time.sleep(0.08)

    def update(self, message: str) -> None:
        """Update the spinner message while running."""
        self._message = message

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.2)
            self._thread = None
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def succeed(self, message: str | None = None) -> None:
        """Stop and show a success message."""
        self.stop()
        sys.stderr.write(f"{GREEN}✔{RESET} {message or self._message}\n")
        sys.stderr.flush()

    def fail(self, message: str | None = None) -> None:
        """Stop and show a failure message."""
        self.stop()
        sys.stderr.write(f"{YELLOW}✖{RESET} {message or self._message}\n")
        sys.stderr.flush()

    @property
    def is_running(self) -> bool:
        return self._running


def truncate(text: str, max_len: int) -> str:
    """Truncate a string to max_len, adding ... if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_params(inp: dict, max_len: int = 80) -> str:
    """Format tool parameters, showing key-value pairs concisely."""
    if not inp:
        return ""

    parts = []
    for k, v in inp.items():
        val = f'"{truncate(v, 40)}"' if isinstance(v, str) else json.dumps(v)
        parts.append(f"{k}: {val}")

    joined = ", ".join(parts)
    return joined if len(joined) <= max_len else joined[: max_len - 3] + "..."


def format_tool_call(name: str, inp: dict) -> str:
    """Format a tool call for display."""
    params = format_params(inp)
    return f"{MAGENTA}🔧 {name}{RESET}{DIM}({params}){RESET}"


def format_duration(ms: float) -> str:
    """Format elapsed time in a human-readable way."""
    if ms < 1000:
        return f"{round(ms)}ms"
    if ms < 60000:
        return f"{ms / 1000:.1f}s"
    return f"{ms / 60000:.1f}m"


def format_tool_result(
    result: str, max_lines: int = 5, max_line_len: int = 120
) -> str:
    """Format a tool result with long output collapsed."""
    lines = result.split("\n")
    total = len(lines)

    shown = [
        line if len(line) <= max_line_len else line[: max_line_len - 3] + "..."
        for line in lines[:max_lines]
    ]

    if total > max_lines:
        shown.append(f"{GRAY}... ({total - max_lines} more lines){RESET}")

    return "\n".join(shown)


def format_tool_cycle(
    name: str, inp: dict, result: str, duration_ms: float
) -> str:
    """Display a complete tool call cycle: name, params, result, duration."""
    header = format_tool_call(name, inp)
    time_str = f"{DIM}[{format_duration(duration_ms)}]{RESET}"
    body = format_tool_result(result)
    return f"{header} {time_str}\n{body}"
