import logging
import math
from typing import Any, Dict, List, Optional

from .base import ConfidenceEvaluator, ConfidenceResult

logger = logging.getLogger(__name__)


def _mean_logprob(token_logprobs: List[float]) -> float:
    """Compute mean of a list of log probabilities, ignoring None values."""
    valid = [lp for lp in token_logprobs if lp is not None]
    if not valid:
        return float("-inf")
    return sum(valid) / len(valid)


def _min_logprob(token_logprobs: List[float]) -> float:
    """Compute min of a list of log probabilities, ignoring None values."""
    valid = [lp for lp in token_logprobs if lp is not None]
    if not valid:
        return float("-inf")
    return min(valid)


class LogProbConfidence(ConfidenceEvaluator):
    """
    Log-probability based confidence.

    Uses the token-level log probabilities of the model's response
    to estimate confidence. Specifically, looks at the logprobs of
    the answer tokens.

    The confidence score is computed as exp(aggregated_logprob) where
    the aggregation can be mean or min over answer tokens.

    Logprobs format expected (OpenAI-compatible):
        Each element in the logprobs list corresponds to one response and
        should be a dict with at least a "token_logprobs" key containing
        a list of float log-probabilities for each generated token.

        Example:
        {
            "tokens": ["The", " answer", " is", " 42"],
            "token_logprobs": [-0.1, -0.05, -0.02, -0.3],
        }
    """

    def __init__(self, aggregation: str = "mean"):
        """
        Args:
            aggregation: How to aggregate token logprobs. One of "mean" or "min".
                - "mean": average logprob across answer tokens (default).
                - "min": minimum logprob across answer tokens (most conservative).
        """
        if aggregation not in ("mean", "min"):
            raise ValueError(f"aggregation must be 'mean' or 'min', got {aggregation!r}")
        self._aggregation = aggregation

    @property
    def name(self) -> str:
        return "logprob"

    @property
    def requires_logprobs(self) -> bool:
        return True

    def evaluate(
        self,
        responses: List[str],
        extracted_answers: List[str],
        question: Optional[str] = None,
        logprobs: Optional[List[Dict]] = None,
    ) -> ConfidenceResult:
        """
        Evaluate confidence using token-level log probabilities.

        For each response, computes an aggregated logprob score over its tokens,
        converts to probability via exp(), and selects the response with the
        highest answer probability.

        Args:
            responses: Raw model outputs (k samples).
            extracted_answers: Post-extraction answers (k samples).
            question: Not used by this method.
            logprobs: Token-level log probabilities for each response.
                Each entry should have a "token_logprobs" key with a list of floats.

        Returns:
            ConfidenceResult where score = exp(aggregated logprob) of the
            best response.
        """
        if not responses:
            return ConfidenceResult(
                score=0.0,
                selected_answer="",
                metadata={"method": "logprob", "error": "no responses provided"},
            )

        if not logprobs or len(logprobs) != len(responses):
            logger.warning(
                "LogProbConfidence: logprobs missing or length mismatch "
                "(responses=%d, logprobs=%s). Returning zero confidence.",
                len(responses),
                len(logprobs) if logprobs else "None",
            )
            return ConfidenceResult(
                score=0.0,
                selected_answer=extracted_answers[0] if extracted_answers else "",
                metadata={
                    "method": "logprob",
                    "error": "logprobs missing or length mismatch",
                },
            )

        # Compute aggregated logprob for each response
        agg_fn = _mean_logprob if self._aggregation == "mean" else _min_logprob
        scores: List[Dict[str, Any]] = []

        for i, lp_data in enumerate(logprobs):
            token_lps = self._extract_token_logprobs(lp_data)
            if not token_lps:
                agg_lp = float("-inf")
            else:
                agg_lp = agg_fn(token_lps)

            # Convert logprob to probability
            prob = math.exp(agg_lp) if agg_lp != float("-inf") else 0.0
            # Clamp to [0, 1]
            prob = max(0.0, min(1.0, prob))

            scores.append(
                {
                    "index": i,
                    "aggregated_logprob": agg_lp,
                    "probability": prob,
                    "num_tokens": len(token_lps),
                }
            )

        # Select the response with the highest probability
        best = max(scores, key=lambda s: s["probability"])
        best_idx = best["index"]

        selected_answer = (
            extracted_answers[best_idx] if best_idx < len(extracted_answers) else ""
        )

        return ConfidenceResult(
            score=best["probability"],
            selected_answer=selected_answer,
            metadata={
                "method": "logprob",
                "aggregation": self._aggregation,
                "best_index": best_idx,
                "best_aggregated_logprob": best["aggregated_logprob"],
                "all_scores": scores,
            },
        )

    @staticmethod
    def _extract_token_logprobs(lp_data: Any) -> List[float]:
        """
        Extract a flat list of token log-probabilities from various formats.

        Supports:
            - Dict with "token_logprobs" key (OpenAI format)
            - Dict with "logprobs" -> "token_logprobs" nested key
            - Plain list of floats
        """
        if lp_data is None:
            return []

        # Plain list of floats
        if isinstance(lp_data, list):
            return [x for x in lp_data if isinstance(x, (int, float))]

        if isinstance(lp_data, dict):
            # Direct "token_logprobs" key
            if "token_logprobs" in lp_data:
                raw = lp_data["token_logprobs"]
                if isinstance(raw, list):
                    return [x for x in raw if isinstance(x, (int, float))]

            # Nested "logprobs" -> "token_logprobs"
            if "logprobs" in lp_data and isinstance(lp_data["logprobs"], dict):
                raw = lp_data["logprobs"].get("token_logprobs", [])
                if isinstance(raw, list):
                    return [x for x in raw if isinstance(x, (int, float))]

            # vLLM / OpenAI ChatCompletion format: "logprobs" -> "content" -> list of token entries
            if "logprobs" in lp_data and isinstance(lp_data["logprobs"], dict):
                content = lp_data["logprobs"].get("content", [])
                if isinstance(content, list):
                    return [
                        entry["logprob"]
                        for entry in content
                        if isinstance(entry, dict) and "logprob" in entry
                    ]

        logger.debug("Could not extract token logprobs from: %s", type(lp_data))
        return []
