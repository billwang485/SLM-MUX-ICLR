import logging
from typing import Any, Dict, List, Optional

from .base import ConfidenceEvaluator, ConfidenceResult

logger = logging.getLogger(__name__)


def _normalize_reward(
    raw_score: float,
    min_score: float = -5.0,
    max_score: float = 5.0,
) -> float:
    """
    Normalize a raw reward score to [0, 1] using linear scaling.

    Args:
        raw_score: The raw reward model output.
        min_score: The minimum expected reward score (maps to 0.0).
        max_score: The maximum expected reward score (maps to 1.0).

    Returns:
        Normalized score clamped to [0, 1].
    """
    if max_score == min_score:
        return 0.5
    normalized = (raw_score - min_score) / (max_score - min_score)
    return max(0.0, min(1.0, normalized))


class RewardModelConfidence(ConfidenceEvaluator):
    """
    Reward model-based confidence.

    Uses a reward model (accessed via OpenAI-compatible API) to score
    each response. The best-scoring response is selected, and its
    reward score (normalized to [0, 1]) is the confidence.

    This can be used with models like:
        - ArmoRM (Nexusflow/Starling-RM-34B)
        - Skywork-Reward
        - Internlm2-reward
        - Any OpenAI-compatible reward model API
    """

    def __init__(
        self,
        provider: Any = None,
        model_name: str = "",
        min_reward: float = -5.0,
        max_reward: float = 5.0,
    ):
        """
        Args:
            provider: An object that can score question-answer pairs.
                Expected interface: provider.score(question, answer, model) -> float
                TODO: Define a formal RewardModelProvider protocol/interface.
            model_name: The reward model identifier.
            min_reward: Minimum expected raw reward score (for normalization).
            max_reward: Maximum expected raw reward score (for normalization).
        """
        self._provider = provider
        self._model_name = model_name
        self._min_reward = min_reward
        self._max_reward = max_reward

    @property
    def name(self) -> str:
        return "reward_model"

    @property
    def requires_external_model(self) -> bool:
        return True

    def evaluate(
        self,
        responses: List[str],
        extracted_answers: List[str],
        question: Optional[str] = None,
        logprobs: Optional[List[Dict]] = None,
    ) -> ConfidenceResult:
        """
        Evaluate confidence using a reward model.

        For each response, obtains a reward score from the provider.
        The response with the highest reward is selected, and its
        normalized reward becomes the confidence score.

        Args:
            responses: Raw model outputs (k samples). Each is scored.
            extracted_answers: Post-extraction answers (k samples).
            question: Original question text (required for reward scoring).
            logprobs: Not used by this method.

        Returns:
            ConfidenceResult with score = normalized reward of best response.
        """
        if not responses:
            return ConfidenceResult(
                score=0.0,
                selected_answer="",
                metadata={"method": "reward_model", "error": "no responses provided"},
            )

        if question is None:
            logger.warning(
                "RewardModelConfidence: question is None. "
                "Most reward models need the question for proper scoring."
            )

        # Score each response
        raw_scores = self._score_responses(responses, question)

        if raw_scores is None:
            return ConfidenceResult(
                score=0.0,
                selected_answer=extracted_answers[0] if extracted_answers else "",
                metadata={
                    "method": "reward_model",
                    "error": "provider call failed or not configured",
                },
            )

        # Find the best response
        best_idx = max(range(len(raw_scores)), key=lambda i: raw_scores[i])
        best_raw = raw_scores[best_idx]
        normalized_score = _normalize_reward(
            best_raw, self._min_reward, self._max_reward
        )

        selected_answer = (
            extracted_answers[best_idx] if best_idx < len(extracted_answers) else ""
        )

        return ConfidenceResult(
            score=normalized_score,
            selected_answer=selected_answer,
            metadata={
                "method": "reward_model",
                "model_name": self._model_name,
                "best_index": best_idx,
                "best_raw_reward": best_raw,
                "all_raw_rewards": raw_scores,
                "all_normalized_rewards": [
                    _normalize_reward(s, self._min_reward, self._max_reward)
                    for s in raw_scores
                ],
            },
        )

    def _score_responses(
        self,
        responses: List[str],
        question: Optional[str],
    ) -> Optional[List[float]]:
        """
        Score all responses using the reward model provider.

        Returns:
            List of raw reward scores, or None if the call fails.
        """
        if self._provider is None:
            logger.warning(
                "RewardModelConfidence: no provider configured. "
                "Set a provider via __init__ to enable reward scoring."
            )
            return None

        try:
            scores = []
            for response in responses:
                # TODO: Adapt to the actual provider interface once finalized.
                # Expected interface: provider.score(question, answer, model) -> float
                # Some providers may support batch scoring for efficiency.
                score = self._provider.score(
                    question=question or "",
                    answer=response,
                    model=self._model_name,
                )
                scores.append(float(score))
            return scores

        except Exception as e:
            logger.error("RewardModelConfidence provider call failed: %s", e)
            return None
