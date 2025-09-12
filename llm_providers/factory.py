from .anthropic_provider import AnthropicProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider

def build(provider: str, model: str, pricing: dict):
    pr = pricing.get(provider, {}).get(model, {"in": 0.0, "out": 0.0})
    if provider == "anthropic":
        return AnthropicProvider(pr["in"], pr["out"])
    if provider == "gemini":
        return GeminiProvider(pr["in"], pr["out"])
    if provider == "openai":
        return OpenAIProvider(pr["in"], pr["out"])
    raise ValueError(f"Unknown provider: {provider}")