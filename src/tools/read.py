from __future__ import annotations
import os
from dataclasses import dataclass
from http.cookiejar import offset_from_tz_string

from src.llm.types import Tool

READ_TOOL_DEFINITION = Tool(
    name="read_file",
    description=(
        "Read the contents of a file. Returns the file content with line numbers. "
        "Use offset and limit to read specific portions of large files."
    ),
    input_schema={
        "type":"object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute or relative path to the file to read",
            },
            "offset": {
                "type": "number",
                "description": "Line number to start reading from (1-based, default: 1)",
            },
            "limit": {
                "type": "number",
                "description": "Maximum number of lines to read (default: all)",
            },
        },
        "required": ["file_path"],
    }
)

@dataclass
class ReadToolInput:
    "Input parameters for read tool."
    file_path: str
    offset: int = 1
    limit: int | None = None


# Maximum file size to read (1MB)
MAX_FILE_SIZE = 1024 * 1024

def _is_binary(data: bytes) -> bool:
    "Check if content is likely binary by looking for null bytes."
    check_lenght = min(len(data), 8192)
    return b"\x00" in data[:check_lenght]

def _format_with_line_numbser(content: str, offset: int) -> str:
    "Format file content with line numbers."
    lines = content.split("\n")
    # remove trailing empty line from splie
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        return ""
    max_line_num = offset + len(lines) - 1
    pad_width = len(str(max_line_num))
    return "\n".join(
        f"{str(offset + i).rjust(pad_width)}\t{line}"
        for i, line in enumerate(lines)
    )

def execute_read_tool(input: ReadToolInput) -> str:
    """Execute the read tool and return file contents with line numbers."""
    file_path = input.file_path
    offset = input.offset
    limit = input.limit

    # validate offset
    if offset < 1:
        return "Error: offset must be >= 1"

    # check file exists
    if not os.path.exists(file_path):
        return f"Error: file not found: {file_path}"

    if not os.path.isfile(file_path):
        return f"Error: not a file: {file_path}"

    # check file size
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        return f"Error: file too large ({file_size} bytes, max {MAX_FILE_SIZE})"

    try:
        with open(file_path, "rb") as f:
            raw = f.read()
    except OSError as e:
        return f"Error: cannot read file: {e}"

    # Binary Check
    if _is_binary(raw):
        return f"Error: binary file detected: {file_path}"

    content = raw.decode("utf-8", errors="replace")
    all_lines = content.split("\n")

    # Apply offset and limit
    start_idx = offset - 1
    end_idx = start_idx + limit if limit is not None else len(all_lines)
    selectd_lines = all_lines[start_idx:end_idx]

    if not selectd_lines:
        return f"(empty: file has {len(all_lines)} lines, offset {offset} is out of range)"

    return _format_with_line_numbser("\n".join(selectd_lines), offset)

