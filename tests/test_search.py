import os
import shutil
import tempfile

import pytest

from src.tools.glob import GLOB_TOOL_DEFINITION, GlobToolInput, execute_glob_tool
from src.tools.grep import GREP_TOOL_DEFINITION, GrepToolInput, execute_grep_tool


@pytest.fixture(autouse=True)
def test_dir():
    """Create a temp directory with sample files."""
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "src", "utils"), exist_ok=True)
    os.makedirs(os.path.join(d, "tests"), exist_ok=True)

    with open(os.path.join(d, "src", "main.ts"), "w") as f:
        f.write('const x = "hello";\nconsole.log(x);\n')
    with open(os.path.join(d, "src", "utils", "helper.ts"), "w") as f:
        f.write("export function add(a: number, b: number) {\n  return a + b;\n}\n")
    with open(os.path.join(d, "src", "utils", "format.py"), "w") as f:
        f.write("def format_name(name: str) -> str:\n    return name.strip()\n")
    with open(os.path.join(d, "tests", "main.test.ts"), "w") as f:
        f.write('import { test } from "vitest";\ntest("works", () => {});\n')
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write("# Project\nThis is a test project.\n")

    yield d
    shutil.rmtree(d)


# ── Glob Tests ──

class TestGlobToolDefinition:
    def test_has_correct_name(self):
        assert GLOB_TOOL_DEFINITION.name == "glob"


class TestExecuteGlobTool:
    def test_find_ts_files(self, test_dir):
        result = execute_glob_tool(GlobToolInput(pattern="*.ts", path=os.path.join(test_dir, "src")))
        assert "main.ts" in result

    def test_find_files_recursively(self, test_dir):
        result = execute_glob_tool(GlobToolInput(pattern="**/*.ts", path=test_dir))
        assert "main.ts" in result
        assert "helper.ts" in result
        assert "main.test.ts" in result

    def test_find_py_files(self, test_dir):
        result = execute_glob_tool(GlobToolInput(pattern="**/*.py", path=test_dir))
        assert "format.py" in result
        assert ".ts" not in result

    def test_no_matches(self, test_dir):
        result = execute_glob_tool(GlobToolInput(pattern="*.xyz", path=test_dir))
        assert "No files matching" in result

    def test_directory_not_found(self):
        result = execute_glob_tool(GlobToolInput(pattern="*.ts", path="/no/such/dir"))
        assert "Error: directory not found" in result


# ── Grep Tests ──

class TestGrepToolDefinition:
    def test_has_correct_name(self):
        assert GREP_TOOL_DEFINITION.name == "grep"


class TestExecuteGrepTool:
    def test_find_pattern(self, test_dir):
        result = execute_grep_tool(GrepToolInput(pattern="hello", path=test_dir))
        assert "main.ts" in result
        assert "hello" in result

    def test_regex_pattern(self, test_dir):
        result = execute_grep_tool(GrepToolInput(pattern=r"function\s+\w+", path=test_dir))
        assert "helper.ts" in result
        assert "add" in result

    def test_include_filter(self, test_dir):
        result = execute_grep_tool(
            GrepToolInput(pattern="return", path=test_dir, include="*.py")
        )
        assert "format.py" in result
        assert ".ts" not in result

    def test_search_single_file(self, test_dir):
        path = os.path.join(test_dir, "src", "main.ts")
        result = execute_grep_tool(GrepToolInput(pattern="console", path=path))
        assert "console.log" in result

    def test_shows_line_numbers(self, test_dir):
        result = execute_grep_tool(GrepToolInput(pattern="console", path=test_dir))
        # Format: file:line: text
        assert ":2:" in result or ":1:" in result

    def test_no_matches(self, test_dir):
        result = execute_grep_tool(
            GrepToolInput(pattern="nonexistent_xyz", path=test_dir)
        )
        assert "No matches" in result

    def test_invalid_regex(self, test_dir):
        result = execute_grep_tool(GrepToolInput(pattern="[invalid", path=test_dir))
        assert "Error: invalid regex" in result