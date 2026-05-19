from src.llm import Message
from src.history import MessageHistory


class TestMessageHistory:
    def test_add_user_and_assistant_messages(self):
        history = MessageHistory()
        history.add_user("Hello")
        history.add_assistant("Hi there!")

        msgs = history.get_messages()
        assert len(msgs) == 2
        assert msgs[0] == Message(role="user", content="Hello")
        assert msgs[1] == Message(role="assistant", content="Hi there!")

    def test_get_messages_returns_copy(self):
        history = MessageHistory()
        history.add_user("Hello")
        msgs = history.get_messages()
        msgs.append(Message(role="user", content="injected"))
        assert len(history.get_messages()) == 1

    def test_get_last_n(self):
        history = MessageHistory()
        history.add_user("Hello")
        history.add_assistant("Hi there!")
        history.add_user("How are you?")
        history.add_assistant("I'm fine, thanks!")
        msgs = history.get_last_n(2)
        assert len(msgs) == 2
        assert msgs[0] == Message(role="user", content="How are you?")
        assert msgs[1] == Message(role="assistant", content="I'm fine, thanks!")

    def test_get_last_n_exceeds_length(self):
        history = MessageHistory()
        history.add_user("Only")
        assert len(history.get_last_n(10)) == 1

    def test_length(self):
        history = MessageHistory()
        assert history.length == 0
        history.add_user("1")
        history.add_assistant("2")
        assert history.length == 2

    def test_clear(self):
        history = MessageHistory()
        history.add_user("1")
        history.add_assistant("2")
        history.clear()
        assert history.length == 0
        assert history.get_messages() == []

    def test_get_last_message(self):
        history = MessageHistory()
        assert history.get_last_message() is None
        history.add_user("1")
        history.add_assistant("2")
        assert history.get_last_message() == Message(role="assistant", content="2")

    def test_remove_last(self):
        history = MessageHistory()
        history.add_user("1")
        history.add_assistant("2")
        removed = history.remove_last_message()
        assert removed == Message(role="assistant", content="2")
        assert history.length == 1

    def test_remove_last_empty(self):
        history = MessageHistory()
        assert history.remove_last_message() is None

    def test_conversation_alternation(self):
        history = MessageHistory()
        history.add_user("Q1")
        history.add_assistant("A1")
        history.add_user("Q2")
        history.add_assistant("A2")

        msgs = history.get_messages()
        roles = [m.role for m in msgs]
        assert roles == ["user", "assistant", "user", "assistant"]