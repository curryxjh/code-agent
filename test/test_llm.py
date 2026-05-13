
@pytest.mark.asyncio
class TestAnthropicProvider:
    async def test_send_message_and_return_response(self):
        with patch("src.llm.anthropic_provider.AsyncAnthropic"):
            provider = AnthropicProvider(AnthropicConfig(api_key="test-key"))