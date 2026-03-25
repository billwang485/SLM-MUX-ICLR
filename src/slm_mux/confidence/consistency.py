import random
from typing import Dict, List, Optional

from .base import ConfidenceEvaluator, ConfidenceResult


class ConsistencyConfidence(ConfidenceEvaluator):
    """
    Consistency-based confidence from the SLM-MUX paper.

    Confidence = frequency of the majority answer among k samples.
    A model that is confident about an answer will reproduce it consistently
    across multiple samples, yielding a high consistency score.

    Algorithm:
        Given k extracted answers from a single model:
        1. Count frequency of each unique answer: f(y) = count(y) / k
        2. The majority answer y* = argmax_y f(y)
        3. Confidence score s = f(y*) = max_count / k
        4. If there's a tie in votes, pick randomly among tied answers
    """

    def __init__(self, ignore_empty: bool = True):
        """
        Args:
            ignore_empty: If True, empty/blank answers are excluded from voting.
        """
        self._ignore_empty = ignore_empty

    def evaluate(
        self,
        responses: List[str],
        extracted_answers: List[str],
        question: Optional[str] = None,
        logprobs: Optional[List[Dict]] = None,
    ) -> ConfidenceResult:
        """
        Evaluate confidence via majority-vote consistency.

        Args:
            responses: Raw model outputs (k samples). Not used directly but
                kept for interface consistency.
            extracted_answers: Post-extraction answers (k samples). These are
                the values that get voted on.
            question: Original question text. Not used by this method.
            logprobs: Not used by this method.

        Returns:
            ConfidenceResult where score = (count of majority answer) / (total votes).
        """
        # Count non-empty answers
        vote_counts: Dict[str, int] = {}
        for ans in extracted_answers:
            if self._ignore_empty and not ans.strip():
                continue
            vote_counts[ans] = vote_counts.get(ans, 0) + 1

        if not vote_counts:
            return ConfidenceResult(
                score=0.0,
                selected_answer="",
                metadata={
                    "vote_counts": {},
                    "total_votes": 0,
                    "max_count": 0,
                    "num_tied": 0,
                    "method": "consistency",
                },
            )

        total_votes = sum(vote_counts.values())
        max_count = max(vote_counts.values())

        # Find all answers with max count (handle ties)
        top_answers = [a for a, c in vote_counts.items() if c == max_count]

        # Break ties randomly
        selected = (
            random.choice(top_answers) if len(top_answers) > 1 else top_answers[0]
        )

        score = max_count / total_votes if total_votes > 0 else 0.0

        return ConfidenceResult(
            score=score,
            selected_answer=selected,
            metadata={
                "vote_counts": vote_counts,
                "total_votes": total_votes,
                "max_count": max_count,
                "num_tied": len(top_answers),
                "method": "consistency",
            },
        )

    @property
    def name(self) -> str:
        return "consistency"
