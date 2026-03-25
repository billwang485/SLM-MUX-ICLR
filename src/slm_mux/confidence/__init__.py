"""
Confidence evaluation module for SLM-MUX.

Provides multiple methods for estimating how confident a model is in its
answer, which is used by the MUX router to select the best model.

Evaluators:
    ConsistencyConfidence   - Majority-vote consistency (paper's core method)
    VerbalizedConfidence    - Model self-reported confidence via follow-up prompt
    LogProbConfidence       - Token log-probability based confidence
    EmbeddingSimilarityConfidence - Embedding cosine similarity for open-ended tasks
    HiddenStateConfidence   - SAPLMA hidden-state probe (EMNLP 2023)
    RewardModelConfidence   - External reward model scoring
    LearnedRouterConfidence - Pre-trained classifier (RouteLLM-style)
"""

from .base import ConfidenceEvaluator, ConfidenceResult
from .consistency import ConsistencyConfidence
from .embedding import EmbeddingSimilarityConfidence
from .hidden_state import HiddenStateConfidence
from .learned_router import LearnedRouterConfidence
from .logprob import LogProbConfidence
from .reward_model import RewardModelConfidence
from .verbalized import VerbalizedConfidence

__all__ = [
    # Base classes
    "ConfidenceEvaluator",
    "ConfidenceResult",
    # Core methods (paper + SAPLMA)
    "ConsistencyConfidence",
    "EmbeddingSimilarityConfidence",
    "HiddenStateConfidence",
    # Additional methods
    "VerbalizedConfidence",
    "LogProbConfidence",
    "RewardModelConfidence",
    "LearnedRouterConfidence",
]
