from src.llm import create_provider, ProviderConfig


async def main():
    provider = create_provider(
        ProviderConfig(
            provider="openai-compatible",
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1",
            model="gpt-3.5-turbo",
        )
    )

    from src.llm.types import Message

    response = await provider.chat(
        messages=[Message(role="user", content="用一句话解释什么是 TypeScript")],
    )

    print("Response: ", response.text)
    print("Stop Reason: ", response.stop_reason)
    print("Usage: ", response.usage)
