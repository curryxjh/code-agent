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

