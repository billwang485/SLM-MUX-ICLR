"""
slm_mux.providers.lmdeploy -- LMDeploy OpenAI-compatible provider.

Routes chat-completion requests to LMDeploy servers registered in a JSON
registry file (typically ``slurm/lmdeploy_servers.json``).  The servers
expose an OpenAI-compatible ``/v1/chat/completions`` endpoint.

This provider is structurally identical to :class:`VLLMProvider` --
the only difference is the default registry file.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx

from slm_mux.providers.base import GenerationResult, ProviderBase
from slm_mux.providers.retry import retry_with_backoff
from slm_mux.providers.rate_limiter import ConcurrencyLimiter
from slm_mux.providers.registry import load_registry

logger = logging.getLogger(__name__)


class LMDeployProvider(ProviderBase):
    """
    Provider for local LMDeploy OpenAI-compatible inference servers.

    On construction, the server registry is loaded from disk (unless a
    pre-loaded *registry* dict is supplied).  Each ``generate`` call
    resolves the target server from the registry and issues a synchronous
    HTTP POST via ``httpx``.

    Parameters
    ----------
    registry_path:
        Path to the JSON registry file.  Defaults to
        ``slurm/lmdeploy_servers.json`` under the project root.
    registry:
        Optional pre-loaded registry dict (overrides *registry_path*).
    max_retries:
        Maximum number of attempts per request.
    request_timeout:
        HTTP timeout in seconds for each request.
    concurrency_limit:
        Optional cap on simultaneous in-flight requests.  ``None``
        disables the concurrency limiter.
    fallback_registry_paths:
        Extra paths to try (in order) if the primary registry is empty or
        missing.
    """

    def __init__(
        self,
        registry_path: Optional[str] = None,
        registry: Optional[Dict[str, Dict[str, Any]]] = None,
        max_retries: int = 3,
        request_timeout: float = 120.0,
        concurrency_limit: Optional[int] = None,
        fallback_registry_paths: Optional[List[str]] = None,
    ) -> None:
        self._registry_path = registry_path
        self._registry_type = "lmdeploy"
        self._max_retries = max(1, max_retries)
        self._request_timeout = request_timeout
        self._concurrency: Optional[ConcurrencyLimiter] = (
            ConcurrencyLimiter(concurrency_limit)
            if concurrency_limit and concurrency_limit > 0
            else None
        )
        self._fallback_paths = fallback_registry_paths or []

        # Load or accept registry.
        if registry is not None:
            self._registry = registry
        else:
            self._registry = self._load_registry_with_fallbacks()

    # ------------------------------------------------------------------
    # ProviderBase interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:  # type: ignore[override]
        return "lmdeploy"

    def is_available(self, model_name: str) -> bool:
        """Return True if *model_name* is present in the loaded registry."""
        if model_name in self._registry:
            return True
        # Try a lazy reload before giving up.
        self._registry = self._load_registry_with_fallbacks()
        return model_name in self._registry

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _load_registry_with_fallbacks(self) -> Dict[str, Dict[str, Any]]:
        """Load registry, trying fallback paths if the primary is empty."""
        reg = load_registry(
            path=self._registry_path, registry_type=self._registry_type
        )
        if reg:
            return reg

        for fb_path in self._fallback_paths:
            logger.info(
                "Primary registry empty; trying fallback: %s", fb_path
            )
            reg = load_registry(
                path=fb_path, registry_type=self._registry_type
            )
            if reg:
                return reg

        logger.warning(
            "No non-empty registry found (primary=%s, fallbacks=%s)",
            self._registry_path,
            self._fallback_paths,
        )
        return {}

    def reload_registry(
        self,
        path: Optional[str] = None,
    ) -> None:
        """
        Reload the server registry from disk.

        Useful after new LMDeploy servers are started or old ones are
        decommissioned.
        """
        self._registry = load_registry(
            path=path or self._registry_path, registry_type=self._registry_type
        )
        logger.info(
            "Reloaded LMDeploy registry (%d model(s))", len(self._registry)
        )

    def _resolve_endpoint(self, model_name: str) -> Tuple[str, str]:
        """
        Return ``(base_url, chat_url)`` for *model_name*.

        Raises :class:`RuntimeError` if the model is not in the registry.
        """
        entry = self._registry.get(model_name)
        if entry is None:
            # Try a registry reload in case servers were registered after init.
            self._registry = self._load_registry_with_fallbacks()
            entry = self._registry.get(model_name)

        if entry is None:
            available = list(self._registry.keys())
            raise RuntimeError(
                f"Model {model_name!r} not found in LMDeploy registry. "
                f"Available models: {available}"
            )

        host = entry["host"]
        port = entry["port"]
        base_url = f"http://{host}:{port}/v1"
        chat_url = f"{base_url}/chat/completions"
        return base_url, chat_url

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
        top_k: Optional[int],
        seed: Optional[int],
        logprobs: bool,
        top_logprobs: Optional[int],
    ) -> GenerationResult:
        """Issue one HTTP POST to the LMDeploy server."""
        _base_url, chat_url = self._resolve_endpoint(model_name)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": float(temperature),
        }
        if isinstance(max_tokens, int) and max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if n > 1:
            payload["n"] = n
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if seed is not None:
            payload["seed"] = int(seed)
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = int(top_logprobs)

        if self._concurrency is not None:
            self._concurrency.acquire()
        try:
            resp = httpx.post(
                chat_url,
                json=payload,
                timeout=self._request_timeout,
            )
        finally:
            if self._concurrency is not None:
                self._concurrency.release()

        if not resp.is_success:
            body = ""
            try:
                body = resp.text
            except Exception:
                pass
            logger.error(
                "[LMDeployProvider] HTTP %d for model=%s url=%s: %s",
                resp.status_code,
                model_name,
                chat_url,
                body[:500],
            )
        resp.raise_for_status()

        data = resp.json()

        # -- Parse content ------------------------------------------------
        content = ""
        raw_logprobs = None
        choices = data.get("choices") or []
        if choices:
            choice0 = choices[0]
            msg = choice0.get("message") or {}
            content = (msg.get("content") or "").strip()
            lp = choice0.get("logprobs")
            if lp is not None:
                raw_logprobs = lp

        # -- Parse usage --------------------------------------------------
        usage_raw = data.get("usage") or {}
        token_usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(
                usage_raw.get("completion_tokens", 0) or 0
            ),
            "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        }

        return GenerationResult(
            content=content,
            reasoning="",
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
        top_k: Optional[int] = None,
        top_logprobs: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Send a chat-completion request to a local LMDeploy server.

        Parameters are documented in :meth:`ProviderBase.generate`.

        Additional keyword-only parameters:

        top_k:
            Top-k sampling parameter.
        top_logprobs:
            Number of top log-probs per token.

        Raises
        ------
        RuntimeError
            If *model_name* is not in the server registry.
        httpx.HTTPStatusError
            If all retry attempts are exhausted.
        """

        @retry_with_backoff(
            max_retries=self._max_retries,
            base_delay=1.0,
            max_delay=30.0,
            retryable_exceptions=(
                httpx.HTTPStatusError,
                httpx.ConnectError,
                httpx.TimeoutException,
                httpx.RemoteProtocolError,
                ConnectionError,
                OSError,
            ),
        )
        def _do_request() -> GenerationResult:
            return self._request_once(
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
            )

        return _do_request()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def available_models(self) -> List[str]:
        """Return a list of model names currently in the registry."""
        return list(self._registry.keys())
