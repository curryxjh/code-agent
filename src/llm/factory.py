from llm.anthropic_provider import AnthropicProvider, AnthropicConfig
from llm.openai_compatible import OpenAICompatibleProvider, OpenAICompatibleConfig
from openai import base_url


@dataclass
class ProviderConfig:
    provider: str # "anthropic" or "openai-compatible"
    api_key: str
    model: str | None=None
    base_url: str | None=None

def create_provider(config: ProviderConfig):
    """Create an LLM provider instance from config."""
    if config.provider == "anthropic":
        return AnthropicProvider(
            AnthropicConfig(
                api_key=config.api_key,
                model=config.model or "claude-sonnet-4-20250514"
            )
        )


    if config.provider == "openai-compatible":
        if not config.base_url:
            raise ValueError("base_url is required for openai-compatible provider")
        if not config.model:
            raise ValueError("model is required for openai-compatible provider")
        return OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model
            ),
        )