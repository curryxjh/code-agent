import re
from src.llm.types import Tool, Message
from src.token_counter import estimate_message_tokens

class Scratchpad:
    """
    Agent Scratchpad — persistent working notes across iterations.

    The scratchpad lets the agent maintain structured notes (plan, findings,
    decisions) that persist across conversation turns. It is injected into
    each LLM call so the agent can stay on track without re-reading the
    entire conversation.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, str]] = []

    def set(self, key: str, value: str) -> None:
        for entry in self._entries:
            if entry["key"] == key:
                entry["value"] = value
                return
        self._entries.append({"key": key, "value": value})

    def get(self, key: str) -> str | None:
        for entry in self._entries:
            if entry["key"] == key:
                return entry["value"]
        return None

    def delete(self, key: str) -> bool:
        for i, entry in enumerate(self._entries):
            if entry["key"] == key:
                self._entries.pop(i)
                return True
        return False

    def has(self, key: str) -> bool:
        return any(e["key"] == key for e in self._entries)

    def clear(self) -> None:
        self._entries.clear()

    def format(self) -> str:
        """Format scratchpad as text to inject into the system prompt."""
        if not self._entries:
            return ""
        lines = [f"- **{e['key']}**: {e['value']}" for e in self._entries]
        return f"## Scratchpad\n" + "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._entries)

# Scratchpad tool definitions for the agent
SCRATCHPAD_TOOLS: list[Tool] = [
    Tool(
        name="scratchpad_set",
        description="Save a note to the scratchpad. Use this to track your plan, findings, or decisions.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Note key (e.g. 'plan', 'findings')"},
                "value": {"type": "string", "description": "Note content"},
            },
            "required": ["key", "value"],
        },
    ),
    Tool(
        name="scratchpad_get",
        description="Read a note from the scratchpad by key.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Note key to read"},
            },
            "required": ["key"],
        },
    ),
    Tool(
        name="scratchpad_list",
        description="List all scratchpad entries.",
        input_schema={"type": "object", "properties": {}},
    ),
]


def execute_scratchpad_tool(
    scratchpad: Scratchpad, name: str, inp: dict
) -> str:
    """Execute a scratchpad tool call."""
    if name == "scratchpad_set":
        key = inp["key"]
        value = inp["value"]
        scratchpad.set(key, value)
        return f'Saved "{key}" to scratchpad.'
    elif name == "scratchpad_get":
        key = inp["key"]
        value = scratchpad.get(key)
        return value if value is not None else f'No entry found for "{key}".'
    elif name == "scratchpad_list":
        return scratchpad.format() or "Scratchpad is empty."
    else:
        return f"Unknown scratchpad tool: {name}"


def select_messages(messages: list[Message], max_tokens: int) -> list[Message]:
    """
    Select messages using a sliding window strategy.

    Keeps the first message (usually the initial user request) and
    the most recent N messages, dropping messages in between when
    the total exceeds the token budget.
    """
    if len(messages) <= 2:
        return list(messages)

    # Always keep the first message
    first = messages[0]
    first_tokens = estimate_message_tokens(first)

    if first_tokens >= max_tokens:
        return [first]

    # Fill from the end
    budget = max_tokens - first_tokens
    tail: list[Message] = []

    for i in range(len(messages) - 1, 0, -1):
        tokens = estimate_message_tokens(messages[i])
        if tokens > budget:
            break
        budget -= tokens
        tail.insert(0, messages[i])

    return [first, *tail]

def detect_context_poisoning(text: str) -> list[str]:
    """
    Detect potential context poisoning — when a tool result contains
    text that looks like it's trying to inject instructions.

    Returns suspicious patterns found, or empty list if clean.
    """
    patterns = [
        (re.compile(r"ignore (?:all )?(?:previous |above )?instructions", re.IGNORECASE), "instruction override"),
        (re.compile(r"you are now", re.IGNORECASE), "role hijacking"),
        (re.compile(r"system:\s", re.IGNORECASE), "system prompt injection"),
        (re.compile(r"\bdo not\b.*\btool", re.IGNORECASE), "tool suppression"),
        (re.compile(r"</?(?:system|instruction|prompt)>", re.IGNORECASE), "fake XML tags"),
    ]

    found: list[str] = []
    for pattern, label in patterns:
        if pattern.search(text):
            found.append(label)
    return found