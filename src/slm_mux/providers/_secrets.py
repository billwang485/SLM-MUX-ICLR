"""
slm_mux.providers._secrets -- Thread-safe API key loading with caching.

Lookup priority:
  1. Environment variable (e.g. ``TOGETHER_API_KEY``).
  2. ``secrets/api_keys.json`` in the project root.
  3. ``$HOME/secrets/api_keys.json``.

Once resolved, the value is cached for the lifetime of the process so
that subsequent calls are essentially free.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal caches
# ---------------------------------------------------------------------------
_cache_lock = threading.Lock()
_key_cache: Dict[str, Optional[str]] = {}
_json_cache: Dict[str, Optional[Dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> str:
    """
    Return the project root directory.

    Heuristic: walk upward from this file until we find a directory that
    contains a ``slurm/`` folder or a ``pyproject.toml`` -- both are
    reliable markers of the SLM-MUX project root.  Falls back to the
    grandparent of the package directory (``src/slm_mux/providers``).
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    # pkg_dir = .../src/slm_mux/providers
    # Walk up at most 6 levels.
    candidate = pkg_dir
    for _ in range(6):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            break
        candidate = parent
        if (
            os.path.isdir(os.path.join(candidate, "slurm"))
            or os.path.isfile(os.path.join(candidate, "pyproject.toml"))
        ):
            return candidate
    # Fallback: three levels up from providers/
    return os.path.dirname(os.path.dirname(os.path.dirname(pkg_dir)))


def _home_dir() -> str:
    return os.path.expanduser("~")


def _read_json_cached(path: str) -> Optional[Dict[str, Any]]:
    """Read and cache a JSON file.  Returns ``None`` on any error."""
    with _cache_lock:
        if path in _json_cache:
            return _json_cache[path]

    data: Optional[Dict[str, Any]] = None
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                data = raw
    except Exception as exc:
        logger.debug("_read_json_cached: failed to read %s: %s", path, exc)

    with _cache_lock:
        _json_cache[path] = data
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_api_key(
    env_name: str,
    json_keys: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Resolve an API key by checking (in order):

    1. The environment variable *env_name*.
    2. Keys listed in *json_keys* inside ``secrets/api_keys.json`` found
       under the project root.
    3. Same file under ``$HOME/secrets/``.

    The result is cached (per *env_name*) after the first successful
    resolution so that file I/O is not repeated.

    Parameters
    ----------
    env_name:
        Name of the environment variable to check first (e.g.
        ``"TOGETHER_API_KEY"``).
    json_keys:
        JSON keys to probe inside ``api_keys.json``.  If ``None``,
        defaults to ``[env_name, env_name.lower()]``.

    Returns
    -------
    The API key string, or ``None`` if not found anywhere.
    """
    # Fast path: check cache.
    with _cache_lock:
        if env_name in _key_cache:
            return _key_cache[env_name]

    if json_keys is None:
        json_keys = [env_name, env_name.lower()]

    # 1. Environment variable.
    val = os.environ.get(env_name)
    if val and val.strip():
        val = val.strip()
        # Some env vars contain comma-separated keys (e.g. Gemini);
        # use only the first one.
        if "," in val:
            val = val.split(",")[0].strip()
        with _cache_lock:
            _key_cache[env_name] = val
        return val

    # 2-3. JSON files.
    search_paths = [
        os.path.join(_project_root(), "secrets", "api_keys.json"),
        os.path.join(_home_dir(), "secrets", "api_keys.json"),
    ]
    for path in search_paths:
        obj = _read_json_cached(path)
        if not obj:
            continue
        for key in json_keys:
            # Try exact, upper, and lower case.
            for variant in (key, key.upper(), key.lower()):
                candidate = obj.get(variant)
                if isinstance(candidate, str) and candidate.strip():
                    resolved = candidate.strip()
                    with _cache_lock:
                        _key_cache[env_name] = resolved
                    return resolved

    # Not found anywhere.
    with _cache_lock:
        _key_cache[env_name] = None
    return None


def invalidate_cache(env_name: Optional[str] = None) -> None:
    """
    Clear the internal key / JSON caches.

    If *env_name* is given, only that entry is evicted; otherwise the
    entire cache is flushed.  Useful in tests or after rotating keys.
    """
    with _cache_lock:
        if env_name is not None:
            _key_cache.pop(env_name, None)
        else:
            _key_cache.clear()
            _json_cache.clear()
