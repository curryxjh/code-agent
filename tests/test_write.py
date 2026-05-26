import os
import shutil
import tempfile

import pytest

from src.tools.write import (
    WRITE_TOOL_DEFINITION,
    WriteToolInput,
    execute_write_tool,
)


@pytest.fixture(autouse=True)
def test_dir():
    """Create and clean up a temp directory."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


class TestWriteToolDefinition:
    def test_has_correct_name(self):
        assert WRITE_TOOL_DEFINITION.name == "write_file"

    def test_has_required_fields(self):
        schema = WRITE_TOOL_DEFINITION.input_schema
        assert "file_path" in schema["required"]
        assert "content" in schema["required"]


class TestExecuteWriteTool:
    def test_create_new_file(self, test_dir):
        path = os.path.join(test_dir, "new.txt")
        result = execute_write_tool(WriteToolInput(file_path=path, content="hello\nworld\n"))
        assert "Created" in result
        assert "new.txt" in result
        with open(path) as f:
            assert f.read() == "hello\nworld\n"

    def test_create_parent_directories(self, test_dir):
        path = os.path.join(test_dir, "deep", "nested", "file.txt")
        result = execute_write_tool(WriteToolInput(file_path=path, content="nested content"))
        assert "Created" in result
        assert os.path.exists(path)

    def test_overwrite_and_show_diff(self, test_dir):
        path = os.path.join(test_dir, "existing.txt")
        with open(path, "w") as f:
            f.write("old line 1\nold line 2\n")
        result = execute_write_tool(
            WriteToolInput(file_path=path, content="new line 1\nold line 2\n")
        )
        assert "Updated" in result
        assert "-old line 1" in result
        assert "+new line 1" in result
        with open(path) as f:
            assert f.read() == "new line 1\nold line 2\n"

    def test_no_changes(self, test_dir):
        path = os.path.join(test_dir, "same.txt")
        with open(path, "w") as f:
            f.write("same content\n")
        result = execute_write_tool(
            WriteToolInput(file_path=path, content="same content\n")
        )
        assert "(no changes)" in result

    def test_reports_line_count(self, test_dir):
        path = os.path.join(test_dir, "lines.txt")
        result = execute_write_tool(
            WriteToolInput(file_path=path, content="a\nb\nc\n")
        )
        assert "4 lines" in result