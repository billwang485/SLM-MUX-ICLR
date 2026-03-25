"""
slm_mux.engine.tie_breaker -- Tie-breaking strategies for SLM-MUX.

When multiple models have the same confidence score, a tie-breaker decides
which model's answer to use as the final prediction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfidence:
    """Confidence result for a single model on a single question."""

    model_name: str
    confidence_score: float
    selected_answer: str
    metadata: Dict = field(default_factory=dict)


class TieBreaker(ABC):
    """Abstract base class for tie-breaking strategies."""

    @abstractmethod
    def break_tie(
        self,
        candidates: List[ModelConfidence],
        model_accuracies: Optional[Dict[str, float]] = None,
    ) -> ModelConfidence:
        """
        Select one winner from a list of tied candidates.

        Args:
            candidates: Models that are tied on confidence score.
                        Should be in model_list order.
            model_accuracies: Optional mapping of model_name -> validation
                              accuracy for accuracy-based tie-breaking.

        Returns:
            The selected ModelConfidence.
        """
        ...


class ValidationAccuracyTieBreaker(TieBreaker):
    """
    Break ties by validation accuracy (the paper's default method).

    Among tied candidates, select the model with the highest known validation
    accuracy.  Falls back to the first candidate if accuracies are unavailable.
    """

    def break_tie(
        self,
        candidates: List[ModelConfidence],
        model_accuracies: Optional[Dict[str, float]] = None,
    ) -> ModelConfidence:
        if not candidates:
            raise ValueError("No candidates to break tie among")
        if len(candidates) == 1:
            return candidates[0]
        if model_accuracies:
            best = max(
                candidates,
                key=lambda c: model_accuracies.get(c.model_name, 0.0),
            )
            return best
        # Fallback: first in list (i.e. model_list order)
        return candidates[0]


class ModelOrderTieBreaker(TieBreaker):
    """
    Break ties by model order in the provided list.

    Candidates are expected to already be sorted in model_list order,
    so the first candidate wins.
    """

    def break_tie(
        self,
        candidates: List[ModelConfidence],
        model_accuracies: Optional[Dict[str, float]] = None,
    ) -> ModelConfidence:
        if not candidates:
            raise ValueError("No candidates to break tie among")
        return candidates[0]


class RandomTieBreaker(TieBreaker):
    """Break ties randomly."""

    def break_tie(
        self,
        candidates: List[ModelConfidence],
        model_accuracies: Optional[Dict[str, float]] = None,
    ) -> ModelConfidence:
        if not candidates:
            raise ValueError("No candidates to break tie among")
        import random

        return random.choice(candidates)
