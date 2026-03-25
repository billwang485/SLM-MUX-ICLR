"""
slm_mux.providers.together -- Together AI provider.

Sends chat-completion requests to the Together REST API
(``https://api.together.xyz/v1/chat/completions``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests

from slm_mux.providers.base import GenerationResult, ProviderBase
from slm_mux.providers.retry import retry_with_backoff
from slm_mux.providers.rate_limiter import QPSLimiter, ConcurrencyLimiter
from slm_mux.providers._secrets import load_api_key

logger = logging.getLogger(__name__)

_TOGETHER_ENDPOINT = "https://api.together.xyz/v1/chat/completions"


def _unify_model_name(model_name: str) -> str:
    """
    Normalise a model identifier for the Together API.

    * Strips replication suffixes (``"model::rep1"`` -> ``"model"``).
    * Replaces underscores with slashes so that shorthand names work
      (e.g. ``"meta-llama_Llama-3.1-8B-Instruct-Turbo"`` ->
      ``"meta-llama/Llama-3.1-8B-Instruct-Turbo"``).
    """
    base = model_name.split("::", 1)[0]
    return base.replace("_", "/")


class TogetherProvider(ProviderBase):
    """
    Provider for the `Together AI <https://api.together.xyz>`_ inference API.

    Parameters
    ----------
    api_key:
        Together API key.  If ``None``, the key is resolved automatically
        from the ``TOGETHER_API_KEY`` environment variable or from
        ``secrets/api_keys.json``.
    max_retries:
        Maximum number of attempts per request (including the initial
        call).
    timeout:
        ``(connect, read)`` timeout in seconds for ``requests.post``.
    qps_limit:
        Optional queries-per-second cap.  ``None`` disables QPS throttling.
    concurrency_limit:
        Optional maximum number of in-flight requests.  ``None`` disables
        the concurrency cap.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        timeout: Tuple[int, int] = (10, 300),
        qps_limit: Optional[float] = None,
        concurrency_limit: Optional[int] = None,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max(1, max_retries)
        self._timeout = timeout
        # Rate limiters are only instantiated when a limit is supplied.
        self._qps: Optional[QPSLimiter] = (
            QPSLimiter(qps_limit) if qps_limit and qps_limit > 0 else None
        )
        self._concurrency: Optional[ConcurrencyLimiter] = (
            ConcurrencyLimiter(concurrency_limit)
            if concurrency_limit and concurrency_limit > 0
            else None
        )

    # ------------------------------------------------------------------
    # ProviderBase interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:  # type: ignore[override]
        return "together"

    def is_available(self, model_name: str) -> bool:
        """Together models are assumed available if an API key is resolvable."""
        try:
            self._resolve_api_key()
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        """Return the API key, raising ``RuntimeError`` if unavailable."""
        if self._api_key:
            return self._api_key
        key = load_api_key(
            "TOGETHER_API_KEY",
            json_keys=["TOGETHER_API_KEY", "together_api_key"],
        )
        if key:
            self._api_key = key
            return key
        raise RuntimeError(
            "Together API key not found. Set the TOGETHER_API_KEY environment "
            "variable or add it to secrets/api_keys.json."
        )

    # ------------------------------------------------------------------
    # Single-attempt request
    # ------------------------------------------------------------------

    def _request_once(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        top_p: Optional[float],
        top_k: Optional[int],
        seed: Optional[int],
    ) -> GenerationResult:
        """Issue one HTTP request and parse the response."""
        api_key = self._resolve_api_key()
        unified_name = _unify_model_name(model_name)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": unified_name,
            "messages": messages,
            "temperature": float(temperature),
        }
        if isinstance(max_tokens, int) and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if seed is not None:
            payload["seed"] = int(seed)

        # Apply rate limiting (only when limiters are configured).
        if self._qps is not None:
            self._qps.acquire()

        if self._concurrency is not None:
            self._concurrency.acquire()
        try:
            resp = requests.post(
                _TOGETHER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        finally:
            if self._concurrency is not None:
                self._concurrency.release()

        # Surface detailed error info before raising.
        if not resp.ok:
            body = ""
            try:
                body = resp.text
            except Exception:
                pass
            logger.error(
                "[TogetherProvider] HTTP %d for model=%s: %s",
                resp.status_code,
                model_name,
                body[:500],
            )
        resp.raise_for_status()

        data = resp.json()

        # -- Parse content ------------------------------------------------
        content = ""
        choices = data.get("choices") or []
        if choices:
            content = (
                choices[0].get("message", {}).get("content", "") or ""
            ).strip()

        # -- Parse usage --------------------------------------------------
        usage_raw = data.get("usage") or {}
        token_usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }

        return GenerationResult(
            content=content,
            reasoning="",
            token_usage=token_usage,
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
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Send a chat-completion request to Together AI with automatic retries.

        Parameters are documented in :meth:`ProviderBase.generate`.

        Additional keyword-only parameters:

        top_k:
            Top-k sampling parameter (supported by Together but not by
            the base :class:`ProviderBase` signature).

        Note
        ----
        ``n > 1`` is accepted for signature compatibility but is not
        currently supported; only the first completion is returned.

        Raises
        ------
        RuntimeError
            If the API key cannot be resolved.
        requests.exceptions.HTTPError
            If all retry attempts are exhausted.
        """
        if n > 1:
            logger.warning(
                "TogetherProvider: n=%d requested but only n=1 is supported; "
                "returning a single completion.",
                n,
            )

        @retry_with_backoff(
            max_retries=self._max_retries,
            base_delay=1.0,
            max_delay=60.0,
            retryable_exceptions=(
                requests.exceptions.RequestException,
                ValueError,
            ),
        )
        def _do_request() -> GenerationResult:
            return self._request_once(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )

        return _do_request()
