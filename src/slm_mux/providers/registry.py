"""
slm_mux.providers.registry -- Server registry loader.

Reads a JSON file that maps model names to ``{"host": ..., "port": ...}``
entries for locally-hosted inference servers (vLLM, lmdeploy, sglang, etc.).

Default registry path: ``slurm/{registry_type}_servers.json`` relative to
the project root.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _project_root() -> str:
    """
    Resolve the SLM-MUX project root directory.

    Walks upward from this file looking for ``slurm/`` or
    ``pyproject.toml`` markers.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
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
    return os.path.dirname(os.path.dirname(os.path.dirname(pkg_dir)))


def _default_registry_path(registry_type: str) -> str:
    """Return the default absolute path for a given registry type."""
    return os.path.join(_project_root(), "slurm", f"{registry_type}_servers.json")


def load_registry(
    path: Optional[str] = None,
    registry_type: str = "vllm",
) -> Dict[str, Dict[str, Any]]:
    """
    Load and return a server registry.

    Parameters
    ----------
    path:
        Explicit path to the JSON file.  If ``None``, the default path
        ``slurm/{registry_type}_servers.json`` is used.
    registry_type:
        Type of server registry (``"vllm"``, ``"lmdeploy"``,
        ``"sglang"``).  Only used when *path* is ``None``.

    Returns
    -------
    A dict mapping model names to their server metadata::

        {
            "meta-llama/Llama-3.1-8B-Instruct": {
                "host": "gpu-server-01.example.com",
                "port": 8001,
                ...
            },
            ...
        }

    An empty dict is returned (with a warning logged) if the file cannot
    be read or is malformed.
    """
    registry_path = path or _default_registry_path(registry_type)

    if not os.path.isfile(registry_path):
        logger.warning(
            "Registry file not found: %s (type=%s)", registry_path, registry_type
        )
        return {}

    try:
        with open(registry_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to parse registry %s: %s", registry_path, exc)
        return {}

    if not isinstance(data, dict):
        logger.error(
            "Registry %s has unexpected top-level type %s (expected dict)",
            registry_path,
            type(data).__name__,
        )
        return {}

    # Validate entries -- keep only those with at least host + port.
    validated: Dict[str, Dict[str, Any]] = {}
    for model_name, entry in data.items():
        if not isinstance(entry, dict):
            logger.debug(
                "Skipping invalid registry entry for %r (not a dict)", model_name
            )
            continue
        host = entry.get("host")
        port = entry.get("port")
        if not host or port is None:
            logger.debug(
                "Skipping incomplete registry entry for %r (missing host/port)",
                model_name,
            )
            continue
        # Normalise types.
        entry_copy = dict(entry)
        entry_copy["host"] = str(host)
        try:
            entry_copy["port"] = int(port)
        except (TypeError, ValueError):
            logger.debug(
                "Skipping registry entry for %r (port %r is not an integer)",
                model_name,
                port,
            )
            continue
        validated[model_name] = entry_copy

    logger.debug(
        "Loaded %d model(s) from registry %s (type=%s)",
        len(validated),
        registry_path,
        registry_type,
    )
    return validated


def resolve_model(
    model_name: str,
    registry: Optional[Dict[str, Dict[str, Any]]] = None,
    path: Optional[str] = None,
    registry_type: str = "vllm",
) -> Optional[Dict[str, Any]]:
    """
    Look up a single model in the registry.

    Parameters
    ----------
    model_name:
        Exact model name to find (e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``).
    registry:
        Pre-loaded registry dict.  If ``None``, the registry is loaded
        from disk using *path* / *registry_type*.
    path:
        Passed to :func:`load_registry` when *registry* is ``None``.
    registry_type:
        Passed to :func:`load_registry` when *registry* is ``None``.

    Returns
    -------
    The entry dict (with at least ``"host"`` and ``"port"``), or ``None``
    if the model is not found.
    """
    if registry is None:
        registry = load_registry(path=path, registry_type=registry_type)
    return registry.get(model_name)
