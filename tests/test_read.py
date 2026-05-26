import os.path
import shutil
import tempfile

import pytest
from src.tools.read import (
    READ_TOOL_DEFINITION,
    ReadToolInput,
    execute_read_tool
)

@pytest.fixture(autouse=True)
def test_dir():
    """Create and clean up a temp directory with test files."""
    d = tempfile.mkdtemp()
    # Create a sample text file with 20 lines
    lines = [f"Line {i + 1}: content" for i in range(20)]
    with open(os.path.join(d, "sample.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # Create a binary file
    with open(os.path.join(d, "binary.bin"), "wb") as f:
        buf = bytearray(100)
        buf[0] = 0x89
        buf[5] = 0x00
        f.write(bytes(buf))
    yield d
    shutil.rmtree(d)


class TestReadToolDefinition:
    def test_has_correct_name(self):
        assert READ_TOOL_DEFINITION.name == "read_file"

    def test_has_required_fields(self):
        assert "file_path" in READ_TOOL_DEFINITION.input_schema["required"]


class TestExecuteReadTool:
    def test_read_entire_file(self, test_dir):
        path = os.path.join(test_dir, "sample.txt")
        result = execute_read_tool(ReadToolInput(file_path=path))
        assert "1\tLine 1: content" in result
        assert "20\tLine 20: content" in result

    def test_offset(self, test_dir):
        path = os.path.join(test_dir, "sample.txt")
        result = execute_read_tool(ReadToolInput(file_path=path, offset=5))
        lines = result.split("\n")
        assert lines[0].strip().startswith("5\tLine 5: content")
        assert "Line 20: content" in result

    def test_limit(self, test_dir):
        path = os.path.join(test_dir, "sample.txt")
        result = execute_read_tool(ReadToolInput(file_path=path, limit=3))
        lines = result.split("\n")
        assert len(lines) == 3
        assert "Line 1: content" in lines[0]
        assert "Line 3: content" in lines[2]

    def test_offset_and_limit(self, test_dir):
        path = os.path.join(test_dir, "sample.txt")
        result = execute_read_tool(ReadToolInput(file_path=path, offset=10, limit=5))
        lines = result.split("\n")
        assert len(lines) == 5
        assert "Line 10: content" in lines[0]
        assert "Line 14: content" in lines[4]

    def test_file_not_found(self):
        result = execute_read_tool(ReadToolInput(file_path="/no/such/file.txt"))
        assert "Error: file not found" in result

    def test_directory_not_file(self, test_dir):
        result = execute_read_tool(ReadToolInput(file_path=test_dir))
        assert "Error: not a file" in result

    def test_binary_file(self, test_dir):
        path = os.path.join(test_dir, "binary.bin")
        result = execute_read_tool(ReadToolInput(file_path=path))
        assert "Error: binary file detected" in result

    def test_out_of_range_offset(self, test_dir):
        path = os.path.join(test_dir, "sample.txt")
        result = execute_read_tool(ReadToolInput(file_path=path, offset=999))
        assert "empty" in result

    def test_invalid_offset(self, test_dir):
        path = os.path.join(test_dir, "sample.txt")
        result = execute_read_tool(ReadToolInput(file_path=path, offset=0))
        assert "Error: offset must be >= 1" in result