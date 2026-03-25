"""
slm_mux.providers -- LLM provider backends for the SLM-MUX system.

Providers are imported lazily to avoid hard dependency on optional packages
(e.g., httpx for vLLM/SGLang, openai for OpenAI). Import specific providers
directly when needed::

    from slm_mux.providers.together import TogetherProvider
    from slm_mux.providers.vllm import VLLMProvider
"""

from slm_mux.providers.base import GenerationResult, ProviderBase
from slm_mux.providers.retry import retry_with_backoff
from slm_mux.providers.rate_limiter import QPSLimiter, RPMLimiter, ConcurrencyLimiter
from slm_mux.providers.registry import load_registry


def __getattr__(name):
    """Lazy import for provider classes to avoid import-time dependency errors."""
    _lazy = {
        "TogetherProvider": "slm_mux.providers.together",
        "OpenAIProvider": "slm_mux.providers.openai_provider",
        "VLLMProvider": "slm_mux.providers.vllm",
        "GeminiProvider": "slm_mux.providers.gemini",
        "HFEndpointProvider": "slm_mux.providers.hf_endpoint",
        "SGLangProvider": "slm_mux.providers.sglang",
        "LMDeployProvider": "slm_mux.providers.lmdeploy",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'slm_mux.providers' has no attribute {name!r}")


__all__ = [
    "GenerationResult",
    "ProviderBase",
    "retry_with_backoff",
    "QPSLimiter",
    "RPMLimiter",
    "ConcurrencyLimiter",
    "TogetherProvider",
    "OpenAIProvider",
    "VLLMProvider",
    "GeminiProvider",
    "HFEndpointProvider",
    "SGLangProvider",
    "LMDeployProvider",
    "load_registry",
]
