"""
slm_mux.engine -- Core SLM-MUX algorithm implementations.
"""

from slm_mux.engine.mux import MUXEngine, MUXItemResult, MUXReport
from slm_mux.engine.offline import OfflineMUX, ModelQuestionData, SimulationResult
from slm_mux.engine.selector import ModelSelector, ComboScore, ComboResult
from slm_mux.engine.tie_breaker import (
    TieBreaker,
    ModelConfidence,
    ValidationAccuracyTieBreaker,
    ModelOrderTieBreaker,
    RandomTieBreaker,
)

__all__ = [
    "MUXEngine",
    "MUXItemResult",
    "MUXReport",
    "OfflineMUX",
    "ModelQuestionData",
    "SimulationResult",
    "ModelSelector",
    "ComboScore",
    "ComboResult",
    "TieBreaker",
    "ModelConfidence",
    "ValidationAccuracyTieBreaker",
    "ModelOrderTieBreaker",
    "RandomTieBreaker",
]
