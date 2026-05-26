from __future__ import annotations
import os
import fnmatch
from dataclasses import dataclass

from src.llm.types import Tool

# Tool definition for LLM
GLOB_TOOL_DEFINITION = Tool(
    name="glob",
    description=(
        "Find files matching a glob-like pattern. "
        "Searches recursively from the given directory. "
        "Returns matching file paths sorted alphabetically."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": 'The pattern to match files against (e.g. "*.ts", "src/**/*.py")',
            },
            "path": {
                "type": "string",
                "description": "The directory to search in (default: current directory)",
            },
        },
        "required": ["pattern"],
    },
)

@dataclass
class GlobToolInput:
    """Input parameters for the glob tool."""
    pattern: str
    path: str = "."

MAX_RESULTS = 200

# Directories to always skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "dist", "build",
    ".next", ".cache", "coverage",
}

def execute_glob_tool(input: GlobToolInput) -> str:
    """Execute the glob tool and return matching file paths."""
    pattern = input.pattern
    search_path = input.path

    if not os.path.exists(search_path):
        return f"Error: directory not found: {search_path}"
    if not os.path.isdir(search_path):
        return f"Error: not a directory: {search_path}"

    results: list[str] = []
    has_double_star = "**" in pattern

    for root, dirs, files in os.walk(search_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for name in files:
            if len(results) >= MAX_RESULTS:
                break
            rel_path = os.path.relpath(os.path.join(root, name), search_path)
            if has_double_star:
                # Match against full relative path
                if fnmatch.fnmatch(rel_path, pattern):
                    results.append(rel_path)
            else:
                # Match against filename only
                if fnmatch.fnmatch(name, pattern):
                    results.append(rel_path)

        if len(results) >= MAX_RESULTS:
            break

    results.sort()

    if not results:
        return f'No files matching "{pattern}" found in {search_path}'

    output = "\n".join(results)

    if len(results) >= MAX_RESULTS:
        output += f"\n\n(showing first {MAX_RESULTS} matches)"

    return output