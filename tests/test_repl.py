from __future__ import annotations

import pytest

from src.repl import (
    Repl,
    ReplConfig,
    Command,
    format_help,
    is_multi_line,
    normalize_input,
    parse_command,
)


class TestNormalizeInput:
    def test_trim_whitespace(self):
        assert normalize_input("  hello  ") == "hello"

    def test_empty_string(self):
        assert normalize_input("") == ""
        assert normalize_input("   ") == ""


class TestParseCommand:
    def test_extract_command_name(self):
        assert parse_command("/help") == "/help"
        assert parse_command("/help arg") == "/help"

    def test_lowercase(self):
        assert parse_command("/HELP") == "/help"

    def test_plain_text(self):
        assert parse_command("hello world") == "hello"


class TestIsMultiLine:
    def test_detect_newlines(self):
        assert is_multi_line("line1\nline2") is True

    def test_single_line(self):
        assert is_multi_line("single line") is False


class TestFormatHelp:
    def test_list_commands(self):
        commands = [
            Command(name="/help", description="Show help", execute=lambda: ""),
            Command(name="/clear", description="Clear screen", execute=lambda: None),
        ]
        help_text = format_help(commands, ["/exit"])
        assert "Available commands:" in help_text
        assert "/help" in help_text
        assert "/clear" in help_text
        assert "/exit" in help_text


class TestRepl:
    @pytest.mark.asyncio
    async def test_blank_input(self):
        repl = Repl()
        assert await repl.process_input("") == ""
        assert await repl.process_input("   ") == ""

    @pytest.mark.asyncio
    async def test_exit_keywords(self):
        repl = Repl()
        assert await repl.process_input("/exit") is None
        assert await repl.process_input("/quit") is None

    @pytest.mark.asyncio
    async def test_custom_exit_keywords(self):
        repl = Repl(ReplConfig(exit_keywords=["/bye"]))
        assert await repl.process_input("/bye") is None
        result = await repl.process_input("/exit")
        assert result is not None

    @pytest.mark.asyncio
    async def test_help_command(self):
        repl = Repl()
        result = await repl.process_input("/help")
        assert result is not None
        assert "Available commands:" in result
        assert "/help" in result

    @pytest.mark.asyncio
    async def test_delegate_to_on_input(self):
        async def handler(text: str) -> str:
            return f"Echo: {text}"

        repl = Repl(ReplConfig(on_input=handler))
        result = await repl.process_input("hello")
        assert result == "Echo: hello"

    @pytest.mark.asyncio
    async def test_unknown_command_without_handler(self):
        repl = Repl()
        result = await repl.process_input("unknown")
        assert result is not None
        assert "Unknown command" in result

    @pytest.mark.asyncio
    async def test_custom_commands(self):
        custom = Command(name="/test", description="Test command", execute=lambda: "test result")
        repl = Repl(ReplConfig(commands=[custom]))
        result = await repl.process_input("/test")
        assert result == "test result"

    @pytest.mark.asyncio
    async def test_custom_commands_in_help(self):
        custom = Command(name="/test", description="Test command", execute=lambda: "ok")
        repl = Repl(ReplConfig(commands=[custom]))
        help_text = await repl.process_input("/help")
        assert help_text is not None
        assert "/test" in help_text
        assert "Test command" in help_text

    @pytest.mark.asyncio
    async def test_case_insensitive(self):
        repl = Repl()
        result = await repl.process_input("/HELP")
        assert result is not None
        assert "Available commands:" in result
        assert await repl.process_input("/EXIT") is None