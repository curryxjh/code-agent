import asyncio
import random
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Awaitable

from src.llm import LLMProvider, ChatOptions, Message, ChatResponse, StreamEvent


@dataclass
class RetryConfig:
    """Retry configuration."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 10.0


DEFAULT_RETRY_CONFIG = RetryConfig()


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate exponential backoff delay with jitter."""
    exponential_delay = config.base_delay * (2 ** attempt)
    capped = min(exponential_delay, config.max_delay)
    # Add jitter: random value between 0 and capped
    return random.random() * capped

def is_retryable(error: Exception) -> bool:
    """Check if an error is retryable (network/server errors)."""

    msg = str(error).lower()

    # Network errors
    if any(kw in msg for kw in ("network", "connection", "timeout")):
        return True

    # Rate limiting
    if "rate limit" in msg or "429" in msg:
        return True

    # Server errors (5xx)
    if any(code in msg for code in ("500", "502", "503")):
        return True

    # Check status attribute
    status = getattr(error, "status", None) or getattr(error, "status_code", None)
    if isinstance(status, int):
        return status >= 500 or status == 429

    return False


class RetryProvider:
    """
    Wraps an LLM provider with automatic retry logic.
    Retries on network errors, rate limits, and server errors.
    """

    def __init__(
        self, provider: LLMProvider, config: RetryConfig | None = None
    ) -> None:
        self._provider = provider
        self._config = config or DEFAULT_RETRY_CONFIG

    async def chat(
            self, messages: list[Message], options: ChatOptions | None = None
    ) -> ChatResponse:
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                return await self._provider.chat(messages, options=options)
            except Exception as e:
                last_error = e
                if not is_retryable(e) or attempt == self._config.max_retries:
                    raise
                delay = calculate_delay(attempt, self._config)
                await asyncio.sleep(delay)

        raise last_error # type: ignore[misc]

    async def stream(
            self, messages: list[Message], options: ChatOptions | None = None
    ) -> AsyncIterator[StreamEvent]:
        last_error: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            try:
                async for event in self._provider.stream(messages, options=options):
                    yield event
                return
            except Exception as e:
                last_error = e
                if not is_retryable(e) or attempt == self._config.max_retries:
                    raise
                delay = calculate_delay(attempt, self._config)
                await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]



def safe_tool_executor(
    executor: Callable[[str, dict], Awaitable[str]],
    known_tools: set[str] | None = None,
) -> Callable[[str, dict], Awaitable[str]]:
    """
    Wraps a tool executor to catch errors and return them as strings
    instead of throwing, so the LLM can see and handle the error.
    """

    async def wrapped(name: str, input: dict) -> str:
        # Check if tool is known
        if known_tools is not None and name not in known_tools:
            available = ", ".join(sorted(known_tools))
            return f'Error: unknown tool "{name}". Available tools: {available}'

        try:
            return await executor(name, input)
        except Exception as e:
            return f"Error executing {name}: {e}"

    return wrapped

