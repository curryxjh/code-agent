from __future__ import annotations
import os
import difflib
from dataclasses import dataclass
from src.llm.types import Tool

WRITE_TOOL_DEFINITION = Tool(
    name="write_file",
    description=(
        "Write content to a file. Creates the file if it doesn't exist, "
        "or overwrites it if it does. Automatically creates parent directories."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute or relative path to the file to write",
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file",
            },
        },
        "required": ["file_path", "content"],
    },
)

@dataclass
class WriteToolInput:
    "Input parameters for the write tool."
    file_path: str
    content: str


def _generate_diff(old_content: str, new_content: str, file_path: str) -> str:
    """Generate a complete diff between old and new content using Myers diff algorithm.
    
    Args:
        old_content: 原始文件内容
        new_content: 新文件内容
        file_path: 文件路径，用于diff头部显示
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = list(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f'a/{file_path}',
        tofile=f'b/{file_path}',
        n=3,
    ))
    
    if not diff:
        return "(no changes)"
    
    return "".join(diff)

def execute_write_tool(input: WriteToolInput) -> str:
    """Execute the write tool and return status with diff."""
    file_path = input.file_path
    content = input.content

    # Read existing content for diff (if file exists)
    old_content: str | None = None
    if os.path.isfile(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                old_content = f.read()
        except OSError:
            pass

    # Create parent directories
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # write file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"Error: cannot write file: {e}"

    lines = len(content.split("\n"))
    if old_content is None:
        return f"Created {file_path} ({lines} lines)"
    diff = _generate_diff(old_content, content, file_path)

    return f"Updated {file_path} ({lines} lines)\n\n{diff}"

