"""
slm_mux.providers.hf_endpoint -- Hugging Face Inference Endpoint provider.

Sends chat-completion requests to a Hugging Face (or AWS-hosted) OpenAI-
compatible inference endpoint.  Merges from both the AWS HF branch of
``utils/api_client.py`` and the standalone ``hf_client.py``.

Key behaviours:
  - System messages are merged into the first user message (some HF/TGI
    endpoints do not support the ``system`` role).
  - The model name sent to the endpoint defaults to ``"tgi"`` for AWS HF
    endpoints.
  - Supports ``n > 1`` for multiple completions in a single request.
  - Longer backoff (60 s) for 5xx server errors.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from slm_mux.providers.base import GenerationResult, ProviderBase
from slm_mux.providers.retry import retry_with_backoff
from slm_mux.providers.rate_limiter import RPMLimiter, ConcurrencyLimiter
from slm_mux.providers._secrets import load_api_key

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Message pre-processing
# ------------------------------------------------------------------

def _prepare_messages_for_hf(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Prepare messages for HF TGI-compatible endpoints.

    * Merge all ``system`` messages into the first ``user`` message
      (creating one if necessary).
    * Normalise unknown roles to ``"user"``.
    """
    if not isinstance(messages, list):
        return []

    system_texts: List[str] = []
    non_system: List[Dict[str, str]] = []

    for msg in messages:
        role = (msg or {}).get("role", "")
        content = (msg or {}).get("content", "")
        if role == "system":
            if content:
                system_texts.append(str(content))
        else:
            if role not in ("user", "assistant"):
                non_system.append({"role": "user", "content": str(content)})
            else:
                non_system.append({"role": role, "content": str(content)})

    if system_texts:
        combined_system = "\n\n".join(system_texts).strip()
        # Find the first user message to prepend system text.
        first_user_idx: Optional[int] = None
        for i, m in enumerate(non_system):
            if m.get("role") == "user":
                first_user_idx = i
                break
        if first_user_idx is not None:
            first_user = non_system[first_user_idx]
            merged = (
                combined_system + "\n\n" + (first_user.get("content") or "")
            ).strip()
            new_first = {"role": "user", "content": merged}
            rest = [m for idx, m in enumerate(non_system) if idx != first_user_idx]
            return [new_first] + rest
        else:
            # No user message exists -- create one with the system text.
            return [{"role": "user", "content": combined_system}] + non_system

    return non_system


# ------------------------------------------------------------------
# Error classification
# ------------------------------------------------------------------

def _is_server_error(exc: Exception) -> bool:
    """Return True if *exc* looks like an HTTP 5xx server error."""
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None)
    if isinstance(status, int) and 500 <= status <= 599:
        return True
    text = str(exc).lower()
    tokens = [
        " 500", " 502", " 503", " 504",
        "server error", "internal server error",
    ]
    return any(tok in text for tok in tokens)


# ------------------------------------------------------------------
# Provider
# ------------------------------------------------------------------

class HFEndpointProvider(ProviderBase):
    """
    Provider for Hugging Face (or AWS-hosted) OpenAI-compatible endpoints.

    Parameters
    ----------
    base_url:
        Base URL of the endpoint (e.g.
        ``"https://…huggingface.cloud/v1/"``).  Falls back to the
        ``HF_ENDPOINT_BASE_URL`` environment variable, then to
        ``"https://…huggingface.cloud"`` hard-coded default.
    api_token:
        HF Bearer token.  If ``None``, resolved from the ``HF_TOKEN``
        environment variable or ``secrets/api_keys.json``.
    model_name_override:
        Model name sent in the request body.  Defaults to ``"tgi"``.
    max_retries:
        Maximum number of attempts per request.
    timeout:
        HTTP timeout in seconds passed to ``requests.post``.
    rpm_limit:
        Optional requests-per-minute cap.  ``None`` disables.
    concurrency_limit:
        Optional maximum in-flight requests.  ``None`` disables.
    """

    _DEFAULT_BASE_URL = ""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        model_name_override: str = "tgi",
        max_retries: int = 6,
        timeout: float = 120.0,
        rpm_limit: Optional[float] = None,
        concurrency_limit: Optional[int] = None,
    ) -> None:
        self._api_token = api_token
        self._model_name_override = model_name_override
        self._max_retries = max(1, max_retries)
        self._timeout = timeout
        self._rpm: Optional[RPMLimiter] = (
            RPMLimiter(rpm_limit) if rpm_limit and rpm_limit > 0 else None
        )
        self._concurrency: Optional[ConcurrencyLimiter] = (
            ConcurrencyLimiter(concurrency_limit)
            if concurrency_limit and concurrency_limit > 0
            else None
        )
        self._session = requests.Session()

        # Resolve base URL.
        raw_url = (
            base_url
            or os.environ.get("HF_ENDPOINT_BASE_URL")
            or self._DEFAULT_BASE_URL
        )
        # Ensure the URL ends with ``/v1`` for OpenAI compatibility.
        if not raw_url.rstrip("/").endswith("/v1"):
            raw_url = raw_url.rstrip("/") + "/v1"
        self._base_url = raw_url

    # ------------------------------------------------------------------
    # ProviderBase interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:  # type: ignore[override]
        return "hf_endpoint"

    def is_available(self, model_name: str) -> bool:
        """Return True if an HF token is resolvable."""
        try:
            self._resolve_api_token()
            return True
        except RuntimeError:
            return False

    # ------------------------------------------------------------------
    # Token resolution
    # ------------------------------------------------------------------

    def _resolve_api_token(self) -> str:
        """Return the HF token, raising ``RuntimeError`` if unavailable."""
        if self._api_token:
            return self._api_token
        key = load_api_key(
            "HF_TOKEN",
            json_keys=["HF_TOKEN", "hf_token", "huggingface"],
        )
        if key:
            self._api_token = key
            return key
        raise RuntimeError(
            "HF token not found. Set the HF_TOKEN environment variable "
            "or add it to secrets/api_keys.json."
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
        n: int,
        top_p: Optional[float],
        top_k: Optional[int],
        seed: Optional[int],
    ) -> GenerationResult:
        """Issue one HTTP POST to the HF endpoint."""
        token = self._resolve_api_token()

        # Prepare messages (merge system role into first user message).
        hf_messages = _prepare_messages_for_hf(messages)

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        # Use the override model name (e.g. "tgi") unless the caller
        # explicitly provides one -- the override is the common case for
        # AWS HF endpoints.
        effective_model = self._model_name_override or model_name

        payload: Dict[str, Any] = {
            "model": effective_model,
            "messages": hf_messages,
            "temperature": float(temperature),
            "stream": False,
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

        # Apply rate limiting.
        if self._rpm is not None:
            self._rpm.acquire()

        if self._concurrency is not None:
            self._concurrency.acquire()
        try:
            resp = self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        finally:
            if self._concurrency is not None:
                self._concurrency.release()

        # Handle vLLM-style logical errors returned as HTTP 200.
        data = resp.json()
        if resp.status_code >= 400 or "error" in data:
            error_detail = data.get("error", {"status": resp.status_code})
            logger.error(
                "[HFEndpointProvider] HTTP %d for model=%s: %s",
                resp.status_code,
                model_name,
                str(error_detail)[:500],
            )
            resp.raise_for_status()

        # -- Parse content ------------------------------------------------
        content = ""
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            content = (msg.get("content") or "").strip()

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
        Send a chat-completion request to a Hugging Face inference endpoint.

        Parameters are documented in :meth:`ProviderBase.generate`.

        Additional keyword-only parameters:

        top_k:
            Top-k sampling parameter.

        Note
        ----
        ``logprobs`` is accepted for signature compatibility but is not
        supported; it is silently ignored.

        The provider uses a longer backoff (60 s) for 5xx server errors
        to allow the HF endpoint to restart.

        Raises
        ------
        RuntimeError
            If the HF token cannot be resolved or a non-recoverable error
            occurs.
        requests.exceptions.HTTPError
            If all retry attempts are exhausted.
        """

        @retry_with_backoff(
            max_retries=self._max_retries,
            base_delay=2.0,
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
                n=n,
                top_p=top_p,
                top_k=top_k,
                seed=seed,
            )

        return _do_request()
