"""
slm_mux.providers.gemini -- Google Gemini REST API provider.

Sends chat-completion requests to the Gemini ``generateContent`` endpoint
(``https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent``).

Ported from the ``_call_gemini_chat()`` function in ``utils/api_client.py``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from slm_mux.providers.base import GenerationResult, ProviderBase
from slm_mux.providers.retry import retry_with_backoff
from slm_mux.providers.rate_limiter import RPMLimiter, ConcurrencyLimiter
from slm_mux.providers._secrets import load_api_key

logger = logging.getLogger(__name__)

_GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
)


# ------------------------------------------------------------------
# Message conversion
# ------------------------------------------------------------------

def _convert_messages_to_gemini(
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Convert OpenAI-style messages into the Gemini ``generateContent`` payload.

    * ``system`` messages are merged into a single ``systemInstruction``.
    * ``user`` / ``assistant`` messages become ``contents`` entries with
      roles ``"user"`` / ``"model"``.
    """
    contents: List[Dict[str, Any]] = []
    system_msgs: List[str] = []

    for msg in messages:
        role = (msg or {}).get("role", "user")
        text = (msg or {}).get("content", "")
        if role == "system":
            if text:
                system_msgs.append(str(text))
            continue
        mapped_role = "model" if role == "assistant" else "user"
        contents.append({"role": mapped_role, "parts": [{"text": str(text)}]})

    # Gemini requires at least one content entry.
    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})

    payload: Dict[str, Any] = {"contents": contents}
    if system_msgs:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_msgs)}]
        }
    return payload


def _redact_api_key(text: str, api_key: Optional[str]) -> str:
    """Prevent accidental logging of sensitive API keys."""
    if not text or not api_key:
        return text
    return text.replace(api_key, "[REDACTED]")


# ------------------------------------------------------------------
# Provider
# ------------------------------------------------------------------

class GeminiProvider(ProviderBase):
    """
    Provider for the `Google Gemini <https://ai.google.dev/>`_ REST API.

    Parameters
    ----------
    api_key:
        Gemini API key.  If ``None``, the key is resolved from the
        ``GEMINI_API_KEY`` environment variable or from
        ``secrets/api_keys.json``.
    max_retries:
        Maximum number of attempts for **non-transient** errors.
        Transient errors (HTTP 429, 5xx, network errors) are retried
        indefinitely with exponential backoff.
    timeout:
        ``(connect, read)`` timeout in seconds for ``requests.post``.
    default_rpm:
        Default requests-per-minute limit.  Override via the
        ``GEMINI_MAX_RPM`` environment variable.  Models whose name
        contains ``"gemini-2.5-pro"`` default to 120 RPM; all others
        default to *default_rpm* (1000).
    concurrency_limit:
        Optional maximum number of in-flight requests.  ``None`` disables.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 5,
        timeout: tuple[int, int] = (10, 300),
        default_rpm: float = 1000.0,
        concurrency_limit: Optional[int] = None,
    ) -> None:
        self._api_key = api_key
        self._max_retries = max(1, max_retries)
        self._timeout = timeout
        self._default_rpm = default_rpm
        self._concurrency: Optional[ConcurrencyLimiter] = (
            ConcurrencyLimiter(concurrency_limit)
            if concurrency_limit and concurrency_limit > 0
            else None
        )
        # RPM limiters are created lazily per effective RPM to handle
        # per-model defaults (gemini-2.5-pro = 120 RPM, others = 1000).
        self._rpm_limiters: Dict[float, RPMLimiter] = {}

    # ------------------------------------------------------------------
    # ProviderBase interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:  # type: ignore[override]
        return "gemini"

    def is_available(self, model_name: str) -> bool:
        """Return True if an API key is resolvable."""
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
            "GEMINI_API_KEY",
            json_keys=["GEMINI_API_KEY", "gemini_api_key", "GOOGLE_API_KEY"],
        )
        if key:
            self._api_key = key
            return key
        raise RuntimeError(
            "Gemini API key not found. Set the GEMINI_API_KEY environment "
            "variable or add it to secrets/api_keys.json."
        )

    # ------------------------------------------------------------------
    # RPM limiter
    # ------------------------------------------------------------------

    def _get_rpm_limiter(self, model_name: str) -> RPMLimiter:
        """
        Return an :class:`RPMLimiter` appropriate for *model_name*.

        The effective RPM is determined by:
          1. ``GEMINI_MAX_RPM`` environment variable (always overrides).
          2. Per-model default (120 for ``gemini-2.5-pro``, else *default_rpm*).
        """
        if "gemini-2.5-pro" in (model_name or ""):
            base_default = 120.0
        else:
            base_default = self._default_rpm

        try:
            env_val = os.environ.get("GEMINI_MAX_RPM", "")
            effective_rpm = float(env_val) if env_val else base_default
        except (ValueError, TypeError):
            effective_rpm = base_default

        if effective_rpm <= 0:
            effective_rpm = base_default

        if effective_rpm not in self._rpm_limiters:
            self._rpm_limiters[effective_rpm] = RPMLimiter(effective_rpm)
        return self._rpm_limiters[effective_rpm]

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
    ) -> GenerationResult:
        """Issue one HTTP POST to the Gemini REST API."""
        api_key = self._resolve_api_key()

        # Build Gemini payload.
        payload = _convert_messages_to_gemini(messages)
        gen_cfg: Dict[str, Any] = {"temperature": float(temperature)}
        if isinstance(max_tokens, int) and max_tokens > 0:
            gen_cfg["maxOutputTokens"] = int(max_tokens)
        if top_p is not None:
            gen_cfg["topP"] = float(top_p)
        if top_k is not None:
            gen_cfg["topK"] = int(top_k)
        payload["generationConfig"] = gen_cfg

        url = f"{_GEMINI_BASE_URL}/{model_name}:generateContent"

        # Apply RPM limiting.
        rpm_limiter = self._get_rpm_limiter(model_name)
        rpm_limiter.acquire()

        # Apply concurrency limiting.
        if self._concurrency is not None:
            self._concurrency.acquire()
        try:
            resp = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=self._timeout,
            )
        finally:
            if self._concurrency is not None:
                self._concurrency.release()

        # Surface error details before raising.
        if not resp.ok:
            body = ""
            try:
                body = resp.text
            except Exception:
                pass
            sanitized = _redact_api_key(
                f"HTTP {resp.status_code} for model={model_name}: {body[:500]}",
                api_key,
            )
            logger.error("[GeminiProvider] %s", sanitized)
        resp.raise_for_status()

        data = resp.json()

        # -- Parse content ------------------------------------------------
        content = ""
        candidates = data.get("candidates") or []
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = [p.get("text", "") for p in parts if p.get("text")]
            content = "".join(texts).strip()

        # -- Parse usage --------------------------------------------------
        usage = data.get("usageMetadata") or {}
        token_usage = {
            "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
            "completion_tokens": int(
                usage.get("candidatesTokenCount", 0) or 0
            ),
            "total_tokens": int(usage.get("totalTokenCount", 0) or 0),
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
        Send a chat-completion request to the Gemini REST API.

        Parameters are documented in :meth:`ProviderBase.generate`.

        Additional keyword-only parameters:

        top_k:
            Top-k sampling parameter (supported by Gemini).

        Note
        ----
        ``n > 1``, ``seed``, and ``logprobs`` are accepted for signature
        compatibility but are not supported by the Gemini REST API; they
        are silently ignored.

        Transient errors (HTTP 429 and 5xx) are retried **indefinitely**
        with exponential backoff.  Non-transient errors give up after
        *max_retries* attempts.

        Raises
        ------
        RuntimeError
            If the API key cannot be resolved or a non-recoverable error
            occurs.
        requests.exceptions.HTTPError
            If all retry attempts for non-transient errors are exhausted.
        """
        if n > 1:
            logger.warning(
                "GeminiProvider: n=%d requested but only n=1 is supported; "
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
            )

        return _do_request()
