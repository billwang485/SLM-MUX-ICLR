"""Base provider interface and common data structures."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationResult:
    """Container for a single generation response from any provider."""

    content: str
    reasoning: str = ""
    token_usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    logprobs: Optional[List[Dict[str, Any]]] = None
    raw_response: Optional[Any] = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self.token_usage.get("total_tokens", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to the legacy dict format used by older SLM-MUX code."""
        d: Dict[str, Any] = {
            "content": self.content,
            "reasoning": self.reasoning,
            "token_usage": dict(self.token_usage),
        }
        if self.logprobs is not None:
            d["logprobs"] = self.logprobs
        return d


class ProviderBase(ABC):
    """Abstract base class that every LLM provider must implement."""

    @abstractmethod
    def generate(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        n: int = 1,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        logprobs: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """Send a chat-completion request and return a GenerationResult."""
        ...

    @abstractmethod
    def is_available(self, model_name: str) -> bool:
        """Return True if this provider can serve the given model right now."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider identifier (e.g. 'together', 'openai')."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider={self.provider_name!r}>"
