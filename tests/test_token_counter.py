
from src.token_counter import (
    estimate_tokens,
    estimate_message_tokens,
    estimate_conversation_tokens,
    get_model_context_limit,
    remaining_budget,
    is_over_budget,
    ContextBudget,
    DEFAULT_BUDGET,
)
from src.llm.types import Message, Tool, TextBlock, ToolUseBlock, ToolResultBlock


class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_english_text(self):
        # "Hello World" = 11 chars -> ceil(11/4) = 3
        assert estimate_tokens("Hello World") == 3

    def test_cjk_text(self):
        # 4 CJK chars -> ceil(4/2) = 2
        assert estimate_tokens("你好世界") == 2

    def test_mixed_cjk_english(self):
        # "Hello你好" = 5 English + 2 CJK -> ceil(5/4) + ceil(2/2) = 2 + 1 = 3
        assert estimate_tokens("Hello你好") == 3

    def test_long_text(self):
        assert estimate_tokens("a" * 1000) == 250

    def test_japanese_text(self):
        tokens = estimate_tokens("こんにちは")
        assert tokens > 0
        assert tokens <= 5


class TestEstimateMessageTokens:
    def test_string_content(self):
        msg = Message(role="user", content="Hello World")
        # 4 (overhead) + 3 (text) = 7
        assert estimate_message_tokens(msg) == 7

    def test_content_blocks(self):
        msg = Message(
            role="assistant",
            content=[TextBlock(text="Hello World")],
        )
        assert estimate_message_tokens(msg) == 7

    def test_tool_use_blocks(self):
        msg = Message(
            role="assistant",
            content=[
                ToolUseBlock(id="123", name="read_file", input={"file_path": "test.txt"}),
            ],
        )
        tokens = estimate_message_tokens(msg)
        assert tokens > 4

    def test_tool_result_blocks(self):
        msg = Message(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="123", content="file contents here"),
            ],
        )
        tokens = estimate_message_tokens(msg)
        assert tokens > 4


class TestEstimateConversationTokens:
    def test_messages_only(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]
        assert estimate_conversation_tokens(messages) > 0

    def test_with_system_prompt(self):
        messages = [Message(role="user", content="Hi")]
        with_system = estimate_conversation_tokens(messages, system="You are helpful.")
        without = estimate_conversation_tokens(messages)
        assert with_system > without

    def test_with_tools(self):
        messages = [Message(role="user", content="Hi")]
        tools = [Tool(name="read_file", description="Read a file", input_schema={"type": "object"})]
        with_tools = estimate_conversation_tokens(messages, tools=tools)
        without = estimate_conversation_tokens(messages)
        assert with_tools > without

    def test_empty_conversation(self):
        assert estimate_conversation_tokens([]) == 0


class TestGetModelContextLimit:
    def test_known_models(self):
        assert get_model_context_limit("deepseek-chat") == 64_000
        assert get_model_context_limit("gpt-4o") == 128_000
        assert get_model_context_limit("claude-sonnet-4-20250514") == 200_000

    def test_unknown_model(self):
        assert get_model_context_limit("unknown-model") is None


class TestContextBudget:
    def test_defaults(self):
        assert DEFAULT_BUDGET.max_context_tokens == 64_000
        assert DEFAULT_BUDGET.reserved_for_response == 4096

    def test_remaining_budget(self):
        budget = ContextBudget(max_context_tokens=10000, reserved_for_response=2000)
        assert remaining_budget(budget, 3000) == 5000

    def test_no_negative_remaining(self):
        budget = ContextBudget(max_context_tokens=1000, reserved_for_response=500)
        assert remaining_budget(budget, 2000) == 0

    def test_over_budget(self):
        budget = ContextBudget(max_context_tokens=10000, reserved_for_response=2000)
        assert is_over_budget(budget, 7000) is False
        assert is_over_budget(budget, 8000) is True
        assert is_over_budget(budget, 9000) is True