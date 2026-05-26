from __future__ import annotations
import fnmatch
import os
import re
from dataclasses import dataclass
from src.llm.types import Tool

#  Tool definition for LLM
GREP_TOOL_DEFINITION = Tool(
name="grep",
    description=(
        "Search file contents for a pattern (regex supported). "
        "Returns matching lines with file paths and line numbers."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search in (default: current directory)",
            },
            "include": {
                "type": "string",
                "description": 'Glob filter for file names (e.g. "*.ts", "*.py")',
            },
        },
        "required": ["pattern"],
    },
)

@dataclass
class GrepToolInput:
    """Input parameters for the grep tool."""
    pattern: str
    path: str = "."
    include: str | None = None


MAX_MATCHES = 100
MAX_FILE_SIZE = 512 * 1024 * 1024 # 512KB



# Directories to skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "dist", "build",
    ".next", ".cache", "coverage",
}

@dataclass
class _GrepMatch:
    file: str
    line: int
    text: str


def _search_file(
        file_path: str, rel_path: str, regex: re.Pattern, matches: list[_GrepMatch]
) -> None:
    """Search a single file for the pattern."""
    try:
        size = os.path.getsize(file_path)
    except OSError:
        return
    if size > MAX_FILE_SIZE:
        return

    try:
        with open(file_path, "rb") as f:
            raw = f.read()
    except OSError:
        return

    # Binary check
    if b"\x00" in raw[:8192]:
        return

    content = raw.decode("utf-8", errors="replace")
    for i, line in enumerate(content.split("\n")):
        if len(matches) >= MAX_MATCHES:
            return
        if regex.search(line):
            matches.append(_GrepMatch(file=rel_path, line=i + 1, text=line))


def execute_grep_tool(input: GrepToolInput) -> str:
    """Execute the grep tool and return matching lines."""
    try:
        regex = re.compile(input.pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    search_path = input.path
    include = input.include

    if not os.path.exists(search_path):
        return f"Error: path not found: {search_path}"

    matches: list[_GrepMatch] = []

    if os.path.isfile(search_path):
        _search_file(search_path, search_path, regex, matches)
    elif os.path.isdir(search_path):
        for root, dirs, files in os.walk(search_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in files:
                if len(matches) >= MAX_MATCHES:
                    break
                if include and not fnmatch.fnmatch(name, include):
                    continue
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, search_path)
                _search_file(full_path, rel_path, regex, matches)
            if len(matches) >= MAX_MATCHES:
                break
    else:
        return f"Error: invalid path: {search_path}"

    if not matches:
        return f'No matches for "{input.pattern}"'

    output = "\n".join(f"{m.file}:{m.line}: {m.text}" for m in matches)
    if len(matches) >= MAX_MATCHES:
        output += f"\n\n(showing first {MAX_MATCHES} matches)"
    return output