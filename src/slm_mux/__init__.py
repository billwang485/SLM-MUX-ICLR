"""slm-mux: Small Language Model Multiplexer.

Multi-model orchestration with confidence-based selection.

Core classes are available via lazy import::

    from slm_mux import MUXEngine, ConsistencyConfidence, load_config
"""

__version__ = "0.1.0"

# Only import truly lightweight base classes eagerly.
# Everything else is lazy to avoid pulling in optional deps (yaml, torch, httpx).
from slm_mux.providers.base import ProviderBase, GenerationResult
from slm_mux.confidence.base import ConfidenceEvaluator, ConfidenceResult
from slm_mux.benchmarks.base import BenchmarkTask, BenchmarkItem


def __getattr__(name):
    _lazy = {
        "MUXEngine": "slm_mux.engine.mux",
        "MUXReport": "slm_mux.engine.mux",
        "OfflineMUX": "slm_mux.engine.offline",
        "ConsistencyConfidence": "slm_mux.confidence.consistency",
        "MUXConfig": "slm_mux.config.schema",
        "load_config": "slm_mux.config.loader",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'slm_mux' has no attribute {name!r}")


__all__ = [
    "ProviderBase",
    "GenerationResult",
    "ConfidenceEvaluator",
    "ConfidenceResult",
    "BenchmarkTask",
    "BenchmarkItem",
    "MUXEngine",
    "MUXReport",
    "OfflineMUX",
    "ConsistencyConfidence",
    "MUXConfig",
    "load_config",
]
