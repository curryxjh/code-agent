"""
Terminal markdown renderer using ANSI escape codes.

Converts a subset of markdown (headings, bold, inline code,
code blocks, lists, horizontal rules) into ANSI-colored output
suitable for terminal display.
"""
from __future__ import annotations

import re

# ANSI escape codes
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
GRAY = "\033[90m"
BG_GRAY = "\033[48;5;236m"
WHITE = "\033[97m"


def render_inline(text: str) -> str:
    """Render inline markdown: **bold**, `code`, *italic*"""
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)
    text = re.sub(r"__(.+?)__", f"{BOLD}\\1{RESET}", text)
    # Inline code: `code`
    text = re.sub(r"`([^`]+)`", f"{CYAN}\\1{RESET}", text)
    # Italic: *text* or _text_
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", f"{ITALIC}\\1{RESET}", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", f"{ITALIC}\\1{RESET}", text)
    return text


def render_code_block(code: str, language: str = "") -> str:
    """Render a code block with optional language label."""
    if language:
        header = f"{GRAY}┌─ {language} {'─' * max(0, 40 - len(language))}┐{RESET}\n"
    else:
        header = f"{GRAY}┌{'─' * 44}┐{RESET}\n"

    lines = [f"{GRAY}│{RESET} {BG_GRAY}{WHITE}{line}{RESET}" for line in code.split("\n")]
    footer = f"\n{GRAY}└{'─' * 44}┘{RESET}"

    return header + "\n".join(lines) + footer


def render_heading(text: str, level: int) -> str:
    """Render a heading (# to ###)."""
    if level == 1:
        prefix = f"{BOLD}{MAGENTA}"
    elif level == 2:
        prefix = f"{BOLD}{GREEN}"
    else:
        prefix = f"{BOLD}{YELLOW}"
    return f"\n{prefix}{'#' * level} {text}{RESET}\n"


def render_list_item(text: str, indent: int = 0) -> str:
    """Render a list item."""
    pad = " " * indent
    return f"{pad}{GREEN}•{RESET} {render_inline(text)}"


def render_horizontal_rule() -> str:
    """Render a horizontal rule."""
    return f"{GRAY}{'─' * 48}{RESET}"


def render_markdown(markdown: str) -> str:
    """Render a complete markdown string for terminal output."""
    lines = markdown.split("\n")
    output: list[str] = []
    in_code_block = False
    code_language = ""
    code_buffer: list[str] = []

    for line in lines:
        # Code block start/end
        if line.lstrip().startswith("```"):
            if in_code_block:
                # End code block
                output.append(render_code_block("\n".join(code_buffer), code_language))
                code_buffer = []
                in_code_block = False
                code_language = ""
            else:
                # Start code block
                in_code_block = True
                code_language = line.lstrip()[3:].strip()
            continue

        if in_code_block:
            code_buffer.append(line)
            continue

        # Horizontal rule
        stripped = line.strip()
        if re.match(r"^---+$", stripped) or re.match(r"^\*\*\*+$", stripped):
            output.append(render_horizontal_rule())
            continue

        # Headings
        heading_match = re.match(r"^(#{1,3})\s+(.+)", line)
        if heading_match:
            output.append(render_heading(heading_match.group(2), len(heading_match.group(1))))
            continue

        # Unordered list items
        list_match = re.match(r"^(\s*)[*-]\s+(.+)", line)
        if list_match:
            output.append(render_list_item(list_match.group(2), len(list_match.group(1))))
            continue

        # Ordered list items
        ordered_match = re.match(r"^(\s*)\d+\.\s+(.+)", line)
        if ordered_match:
            output.append(render_list_item(ordered_match.group(2), len(ordered_match.group(1))))
            continue

        # Empty line
        if stripped == "":
            output.append("")
            continue

        # Regular paragraph text
        output.append(render_inline(line))

    # Handle unclosed code block
    if in_code_block and code_buffer:
        output.append(render_code_block("\n".join(code_buffer), code_language))

    return "\n".join(output)


def strip_ansi(text: str) -> str:
    """Strip all ANSI escape codes from a string (useful for testing)."""
    return re.sub(r"\033\[[0-9;]*[a-zA-Z]", "", text)