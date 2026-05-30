from __future__ import annotations

import time
from unittest.mock import patch

from src.tool_display import (
    Spinner,
    truncate,
    format_params,
    format_tool_call,
    format_duration,
    format_tool_result,
    format_tool_cycle,
)
from src.markdown import strip_ansi


class TestTruncate:
    def test_short_text(self):
        assert truncate("hello", 10) == "hello"

    def test_long_text(self):
        assert truncate("hello world", 8) == "hello..."

    def test_exact_length(self):
        assert truncate("hello", 5) == "hello"


class TestFormatParams:
    def test_key_value_pairs(self):
        result = format_params({"file_path": "test.txt", "limit": 10})
        assert "file_path:" in result
        assert '"test.txt"' in result
        assert "limit: 10" in result

    def test_truncate_long_strings(self):
        result = format_params({"content": "x" * 100})
        assert "..." in result

    def test_empty_input(self):
        assert format_params({}) == ""

    def test_truncate_overall(self):
        inp = {f"key{i}": "value" for i in range(10)}
        result = format_params(inp, 50)
        assert len(result) <= 50


class TestFormatToolCall:
    def test_name_and_params(self):
        result = strip_ansi(format_tool_call("read_file", {"file_path": "test.txt"}))
        assert "read_file" in result
        assert "test.txt" in result

    def test_empty_params(self):
        result = strip_ansi(format_tool_call("list_tasks", {}))
        assert "list_tasks" in result


class TestFormatDuration:
    def test_milliseconds(self):
        assert format_duration(50) == "50ms"
        assert format_duration(999) == "999ms"

    def test_seconds(self):
        assert format_duration(1500) == "1.5s"
        assert format_duration(30000) == "30.0s"

    def test_minutes(self):
        assert format_duration(90000) == "1.5m"


class TestFormatToolResult:
    def test_short_results(self):
        result = format_tool_result("line 1\nline 2")
        assert "line 1" in result
        assert "line 2" in result

    def test_collapse_long_results(self):
        lines = "\n".join(f"line {i}" for i in range(20))
        result = format_tool_result(lines, max_lines=5)
        assert "line 0" in result
        assert "line 4" in result
        assert "15 more lines" in result

    def test_truncate_long_lines(self):
        long_line = "x" * 200
        result = format_tool_result(long_line, max_lines=5, max_line_len=50)
        assert len(strip_ansi(result)) < 200

    def test_empty_result(self):
        assert format_tool_result("") == ""


class TestFormatToolCycle:
    def test_combines_all(self):
        result = strip_ansi(
            format_tool_cycle("read_file", {"file_path": "test.txt"}, "file contents", 150)
        )
        assert "read_file" in result
        assert "test.txt" in result
        assert "150ms" in result
        assert "file contents" in result


class TestSpinner:
    def test_store_message(self):
        spinner = Spinner("Loading...")
        assert spinner.message == "Loading..."

    def test_update_message(self):
        spinner = Spinner("Loading...")
        spinner.update("Still loading...")
        assert spinner.message == "Still loading..."

    def test_current_frame(self):
        spinner = Spinner("test")
        assert spinner.current_frame() == "⠋"

    def test_running_state(self):
        spinner = Spinner("test")
        assert spinner.is_running is False

    def test_start_and_stop(self):
        with patch("sys.stderr"):
            spinner = Spinner("test")
            spinner.start()
            assert spinner.is_running is True
            time.sleep(0.1)
            spinner.stop()
            assert spinner.is_running is False

    def test_succeed_message(self):
        with patch("sys.stderr") as mock_stderr:
            spinner = Spinner("test")
            spinner.succeed("Done!")
            calls = "".join(str(c) for c in mock_stderr.write.call_args_list)
            assert "Done!" in calls

    def test_fail_message(self):
        with patch("sys.stderr") as mock_stderr:
            spinner = Spinner("test")
            spinner.fail("Failed!")
            calls = "".join(str(c) for c in mock_stderr.write.call_args_list)
            assert "Failed!" in calls