from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.errors import (
    RetryConfig,
    RetryProvider,
    calculate_delay,
    is_retryable,
    safe_tool_executor,
)
from src.llm.types import ChatResponse, TextBlock


class TestIsRetryable:
    def test_retry_on_network_errors(self):
        assert is_retryable(Exception("network error")) is True
        assert is_retryable(Exception("connection reset")) is True
        assert is_retryable(Exception("request timeout")) is True

    def test_retry_on_rate_limit(self):
        assert is_retryable(Exception("rate limit exceeded")) is True
        assert is_retryable(Exception("429 Too Many Requests")) is True

    def test_retry_on_server_errors(self):
        assert is_retryable(Exception("500 Internal Server Error")) is True
        assert is_retryable(Exception("502 Bad Gateway")) is True
        assert is_retryable(Exception("503 Service Unavailable")) is True

    def test_retry_on_status_500(self):
        err = Exception("server")
        err.status = 500  # type: ignore[attr-defined]
        assert is_retryable(err) is True

    def test_retry_on_status_429(self):
        err = Exception("limited")
        err.status = 429  # type: ignore[attr-defined]
        assert is_retryable(err) is True

    def test_no_retry_on_client_errors(self):
        assert is_retryable(Exception("invalid api key")) is False
        assert is_retryable(Exception("400 Bad Request")) is False

    def test_no_retry_on_status_400(self):
        err = Exception("bad")
        err.status = 400  # type: ignore[attr-defined]
        assert is_retryable(err) is False


class TestCalculateDelay:
    def test_value_within_range(self):
        config = RetryConfig(max_retries=3, base_delay=0.1, max_delay=5.0)
        for _ in range(100):
            delay = calculate_delay(0, config)
            assert 0 <= delay <= 0.1  # 0.1 * 2^0 = 0.1

    def test_cap_at_max_delay(self):
        config = RetryConfig(max_retries=3, base_delay=1.0, max_delay=2.0)
        for _ in range(100):
            delay = calculate_delay(10, config)
            assert delay <= 2.0


SUCCESS_RESPONSE = ChatResponse(
    content=[TextBlock(type="text", text="OK")],
    text="OK",
    stop_reason="end_turn",
    usage={"input_tokens": 5, "output_tokens": 3},
)


class TestRetryProvider:
    @pytest.mark.asyncio
    async def test_return_on_first_success(self):
        inner = AsyncMock()
        inner.chat = AsyncMock(return_value=SUCCESS_RESPONSE)

        provider = RetryProvider(
            inner, RetryConfig(max_retries=3, base_delay=0.001, max_delay=0.001)
        )
        result = await provider.chat([])

        assert result.text == "OK"
        assert inner.chat.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_and_succeed(self):
        call_count = 0

        async def chat_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("503 Service Unavailable")
            return SUCCESS_RESPONSE

        inner = AsyncMock()
        inner.chat = AsyncMock(side_effect=chat_side_effect)

        provider = RetryProvider(
            inner, RetryConfig(max_retries=3, base_delay=0.001, max_delay=0.001)
        )
        result = await provider.chat([])

        assert result.text == "OK"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_throw_after_max_retries(self):
        inner = AsyncMock()
        inner.chat = AsyncMock(
            side_effect=Exception("500 Internal Server Error")
        )

        provider = RetryProvider(
            inner, RetryConfig(max_retries=2, base_delay=0.001, max_delay=0.001)
        )

        with pytest.raises(Exception, match="500"):
            await provider.chat([])

        assert inner.chat.call_count == 3  # 1 initial + 2 retries

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable(self):
        inner = AsyncMock()
        inner.chat = AsyncMock(side_effect=Exception("invalid api key"))

        provider = RetryProvider(
            inner, RetryConfig(max_retries=3, base_delay=0.001, max_delay=0.001)
        )

        with pytest.raises(Exception, match="invalid api key"):
            await provider.chat([])

        assert inner.chat.call_count == 1


class TestSafeToolExecutor:
    @pytest.mark.asyncio
    async def test_return_result_on_success(self):
        async def executor(name: str, inp: dict) -> str:
            return f"result from {name}"

        safe = safe_tool_executor(executor)
        result = await safe("read_file", {"file_path": "test.txt"})
        assert result == "result from read_file"

    @pytest.mark.asyncio
    async def test_catch_errors(self):
        async def executor(name: str, inp: dict) -> str:
            raise RuntimeError("file not found")

        safe = safe_tool_executor(executor)
        result = await safe("read_file", {"file_path": "missing.txt"})
        assert "Error executing read_file" in result
        assert "file not found" in result

    @pytest.mark.asyncio
    async def test_reject_unknown_tools(self):
        async def executor(name: str, inp: dict) -> str:
            return "result"

        safe = safe_tool_executor(executor, known_tools={"read_file", "write_file"})
        result = await safe("delete_file", {})
        assert 'unknown tool "delete_file"' in result
        assert "read_file" in result

    @pytest.mark.asyncio
    async def test_allow_known_tools(self):
        async def executor(name: str, inp: dict) -> str:
            return "ok"

        safe = safe_tool_executor(executor, known_tools={"read_file"})
        result = await safe("read_file", {})
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_no_check_when_known_tools_not_set(self):
        async def executor(name: str, inp: dict) -> str:
            return "ok"

        safe = safe_tool_executor(executor)
        result = await safe("any_tool", {})
        assert result == "ok"