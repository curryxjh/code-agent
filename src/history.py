from src.llm import Message


class MessageHistory:
    """Manage conversation history between user and assistant."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add_user(self, content: str) -> None:
        self._messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(Message(role="assistant", content=content))

    def get_messages(self) -> list[Message]:
        """Return a shallow copy to protect internal state."""
        return list(self._messages)

    def get_last_n(self, n: int) -> list[Message]:
        return self._messages[-n:]

    @property
    def length(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def get_last_message(self) -> Message | None:
        return self._messages[-1] if self._messages else None

    def remove_last_message(self) -> Message | None:
        return self._messages.pop() if self._messages else None