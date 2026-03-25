from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConfidenceResult:
    """Result of confidence evaluation for a single model on a single question."""

    score: float  # 0.0 to 1.0
    selected_answer: str  # The answer chosen by this evaluator
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfidenceEvaluator(ABC):
    """Base class for all confidence evaluation methods."""

    @abstractmethod
    def evaluate(
        self,
        responses: List[str],
        extracted_answers: List[str],
        question: Optional[str] = None,
        logprobs: Optional[List[Dict]] = None,
    ) -> ConfidenceResult:
        """
        Evaluate confidence given a set of responses from a single model.

        Args:
            responses: Raw model outputs (k samples)
            extracted_answers: Post-extraction answers (k samples)
            question: Original question text (needed for some methods)
            logprobs: Token-level log probabilities (needed for logprob method)

        Returns:
            ConfidenceResult with score, selected answer, and metadata
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def requires_logprobs(self) -> bool:
        return False

    @property
    def requires_external_model(self) -> bool:
        return False
