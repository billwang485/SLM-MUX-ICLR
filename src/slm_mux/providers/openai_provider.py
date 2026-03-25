"""
slm_mux.providers.openai_provider -- OpenAI / OpenAI-compatible provider.

Uses the ``openai`` Python package to call OpenAI (or any compatible
endpoint such as DeepSeek, VectorEngine, etc.) via the chat completions
API.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from slm_mux.providers.base import GenerationResult, ProviderBase
from slm_mux.providers.retry import retry_with_backoff
from slm_mux.providers.rate_limiter import QPSLimiter, RPMLimiter, ConcurrencyLimiter
from slm_mux.providers._secrets import load_api_key

logger = logging.getLogger(__name__)

# Lazy import: the openai package is optional -- we only fail when
# someone actually tries to use this provider without it installed.
_openai_module = None


def _get_openai():
    """Lazily import and return the ``openai.OpenAI`` class."""
    global _openai_module
    if _openai_module is not None:
        return _openai_module
    try:
        from openai import OpenAI  # type: ignore
        _openai_module = OpenAI
        return _openai_module
    except ImportError:
        raise ImportError(
            "The 'openai' package is required for OpenAIProvider. "
            "Install it with: pip install openai"
        )


class OpenAIProvider(ProviderBase):
    """
    Provider for OpenAI and OpenAI-compatible APIs.

    Supports:
      - Standard OpenAI chat completions.
      - Custom base URLs (e.g. DeepSeek, VectorEngine) via *base_url* or
        the ``OPENAI_BASE_URL`` environment variable.
      - ``reasoning_content`` extraction (DeepSeek R1 style).
      - ``logprobs`` passthrough.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Resolved from ``OPENAI_API_KEY`` / secrets if
        not provided.
    base_url:
        Optional custom base URL.  Falls back to ``OPENAI_BASE_URL``.
    max_retries:
        Maximum number of attempts per request.
    timeout:
        ``(connect, read)`` timeout tuple or a single timeout float
        passed to the OpenAI client.
    qps_limit:
        Optional queries-per-second cap.  ``None`` disables.
    rpm_limit:
        Optional requests-per-minute cap.  ``None`` disables.
    concurrency_limit:
        Optional maximum number of in-flight requests.  ``None`` disables.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        timeout: Optional[Any] = None,
        qps_limit: Optional[float] = None,
        rpm_limit: Optional[float] = None,
        concurrency_limit: Optional[int] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max(1, max_retries)
        self._timeout = timeout
        # Rate limiters are only created when a positive limit is provided.
        self._qps: Optional[QPSLimiter] = (
            QPSLimiter(qps_limit) if qps_limit and qps_limit > 0 else None
        )
        self._rpm: Optional[RPMLimiter] = (
            RPMLimiter(rpm_limit) if rpm_limit and rpm_limit > 0 else None
        )
        self._concurrency: Optional[ConcurrencyLimiter] = (
            ConcurrencyLimiter(concurrency_limit)
            if concurrency_limit and concurrency_limit > 0
            else None
        )
        self._client: Any = None  # Lazily initialised OpenAI client.

    # ------------------------------------------------------------------
    # ProviderBase interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:  # type: ignore[override]
        return "openai"

    def is_available(self, model_name: str) -> bool:
        """Return True if an API key is resolvable."""
        try:
            self._resolve_api_key()
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------
    # Key / client resolution
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = load_api_key(
            "OPENAI_API_KEY",
            json_keys=["OPENAI_API_KEY", "openai_api_key"],
        )
        if key:
            self._api_key = key
            return key
        raise RuntimeError(
            "OpenAI API key not found. Set the OPENAI_API_KEY environment "
            "variable or add it to secrets/api_keys.json."
        )

    def _resolve_base_url(self) -> Optional[str]:
        if self._base_url:
            return self._base_url
        return os.environ.get("OPENAI_BASE_URL") or None

    def _get_client(self) -> Any:
        """Return (and cache) the ``openai.OpenAI`` client instance."""
        if self._client is not None:
            return self._client
        OpenAI = _get_openai()
        api_key = self._resolve_api_key()
        base_url = self._resolve_base_url()
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------
    # Single-attempt request
    # ------------------------------------------------------------------

    def _request_once(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        n: int,
        top_p: Optional[float],
        seed: Optional[int],
        logprobs: bool,
        top_logprobs: Optional[int],
    ) -> GenerationResult:
        """Issue one API call and return a parsed result."""
        client = self._get_client()

        call_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": float(temperature),
        }
        if isinstance(max_tokens, int) and max_tokens > 0:
            call_kwargs["max_tokens"] = max_tokens
        if n > 1:
            call_kwargs["n"] = n
        if top_p is not None:
            call_kwargs["top_p"] = float(top_p)
        if seed is not None:
            call_kwargs["seed"] = int(seed)
        if self._timeout is not None:
            call_kwargs["timeout"] = self._timeout
        if logprobs:
            call_kwargs["logprobs"] = True
            if top_logprobs is not None:
                call_kwargs["top_logprobs"] = int(top_logprobs)

        # Apply rate limiting.
        if self._qps is not None:
            self._qps.acquire()
        if self._rpm is not None:
            self._rpm.acquire()

        if self._concurrency is not None:
            self._concurrency.acquire()
        try:
            resp = client.chat.completions.create(**call_kwargs)
        finally:
            if self._concurrency is not None:
                self._concurrency.release()

        # -- Parse content & reasoning ------------------------------------
        content = ""
        reasoning = ""
        raw_logprobs = None

        if getattr(resp, "choices", None):
            choice0 = resp.choices[0]
            msg_obj = getattr(choice0, "message", None)
            if msg_obj is not None:
                content = (getattr(msg_obj, "content", "") or "").strip()
                # DeepSeek R1 / VectorEngine style reasoning.
                reasoning = (
                    getattr(msg_obj, "reasoning_content", "") or ""
                ).strip()
            # Logprobs (if requested).
            lp = getattr(choice0, "logprobs", None)
            if lp is not None:
                raw_logprobs = lp

        # -- Parse usage --------------------------------------------------
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        usage = getattr(resp, "usage", None)
        if usage is not None:
            try:
                token_usage["prompt_tokens"] = int(
                    getattr(usage, "prompt_tokens", 0) or 0
                )
                token_usage["completion_tokens"] = int(
                    getattr(usage, "completion_tokens", 0) or 0
                )
                token_usage["total_tokens"] = int(
                    getattr(usage, "total_tokens", 0) or 0
                )
            except (TypeError, ValueError):
                pass

        return GenerationResult(
            content=content,
            reasoning=reasoning,
            token_usage=token_usage,
            logprobs=raw_logprobs,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
        *,
        top_logprobs: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Send a chat-completion request via the OpenAI Python SDK.

        Parameters are documented in :meth:`ProviderBase.generate`.

        Extra keyword-only parameters:

        top_logprobs:
            Number of top log-probs per token (requires ``logprobs=True``).

        Raises
        ------
        RuntimeError
            If the API key cannot be resolved.
        ImportError
            If the ``openai`` package is not installed.
        """

        @retry_with_backoff(
            max_retries=self._max_retries,
            base_delay=1.0,
            max_delay=60.0,
        )
        def _do_request() -> GenerationResult:
            return self._request_once(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                top_p=top_p,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )

        return _do_request()
