"""
slm_mux.engine.selector -- Model combination search for SLM-MUX.

Port of ``offline_confacc_select.py``.  Searches for optimal model
combinations using the objective:

    O(S) = UnionAcc(S) - lambda * Contradiction(S)

where:
  - UnionAcc(S)     = fraction of questions where at least one model in S is
                       correct (by robust_level criterion).
  - Contradiction(S) = among those "union correct" questions, the fraction
                       where some model is *confidently wrong* (all k samples
                       agree on a wrong answer).
"""

import itertools
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from slm_mux.engine.offline import ModelQuestionData

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Result data classes
# -----------------------------------------------------------------------

@dataclass
class ComboScore:
    """Score breakdown for a single model combination."""

    score: float  # O(S) = union_acc - lambda_c * contradiction_rate
    union_acc: float  # fraction of questions with at least one correct model
    union_correct: int  # count of questions with at least one correct model
    union_total: int  # total number of questions
    contradiction_count: int  # questions where a model is confidently wrong


@dataclass
class ComboResult:
    """Best model combination for a given size k."""

    combo: List[str]
    score: float
    union_acc: float
    union_correct: int
    union_total: int
    contradiction_count: int


# -----------------------------------------------------------------------
# ModelSelector
# -----------------------------------------------------------------------

class ModelSelector:
    """
    Search for optimal model combinations using the confident-accuracy
    objective from the paper.

    The objective penalises model sets that contain a model which is
    *confidently wrong* on questions that the set can otherwise answer
    correctly.  A model is "confidently wrong" on a question when all
    k of its samples agree on the same (incorrect) answer, meaning it
    would win a confidence vote despite being wrong.

    Args:
        lambda_c: Trade-off weight for the contradiction penalty.
        robust_level: Minimum number of correct samples a model must have
            for a question to count as "correct" for that model.
    """

    def __init__(self, lambda_c: float = 0.5, robust_level: int = 2):
        self.lambda_c = lambda_c
        self.robust_level = robust_level

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def compute_score(
        self,
        combo: List[str],
        model_data: Dict[str, Dict[str, ModelQuestionData]],
    ) -> ComboScore:
        """
        Compute O(S) for a given model combination.

        This is a direct port of ``compute_confacc_for_combo()`` from the
        existing ``offline_confacc_select.py``.

        For each question in the union of all models' question sets:
          - ``any_correct``: True if at least one model in the combo has
            >= robust_level correct samples (i.e. samples matching reference).
          - ``any_confident_wrong``: True if some model has > 1 sample,
            all samples agree (unique answer set size == 1), and none of
            them are correct (num_correct == 0).

        Args:
            combo: List of model names forming the combination.
            model_data: model_name -> {question_text -> ModelQuestionData}.

        Returns:
            ComboScore with the full breakdown.
        """
        # Collect the union of all questions across models in the combo
        union_questions: Set[str] = set()
        for m in combo:
            union_questions.update(model_data[m].keys())
        if not union_questions:
            return ComboScore(
                score=0.0,
                union_acc=0.0,
                union_correct=0,
                union_total=0,
                contradiction_count=0,
            )

        union_correct = 0
        contra_in_correct = 0

        for q in union_questions:
            any_correct = False
            any_confident_wrong = False

            for m in combo:
                entry = model_data[m].get(q)
                if entry is None:
                    continue

                # Count how many samples are correct
                num_correct = sum(
                    1 for a in entry.samples if a == entry.reference
                )

                if num_correct >= self.robust_level:
                    any_correct = True

                # Check if the model is confidently wrong: all samples agree
                # on a wrong answer (len > 1 so it's meaningful, all same,
                # none correct)
                if len(entry.samples) > 1:
                    unique_answers = set(entry.samples)
                    if len(unique_answers) == 1 and num_correct == 0:
                        any_confident_wrong = True

            if any_correct:
                union_correct += 1
                # Contradiction only matters when combo has > 1 model
                if any_confident_wrong and len(combo) > 1:
                    contra_in_correct += 1

        union_total = len(union_questions)
        union_acc = union_correct / union_total if union_total else 0.0
        contra_rate = (
            contra_in_correct / union_correct if union_correct else 0.0
        )
        score = union_acc - self.lambda_c * contra_rate

        return ComboScore(
            score=score,
            union_acc=union_acc,
            union_correct=union_correct,
            union_total=union_total,
            contradiction_count=contra_in_correct,
        )

    # ------------------------------------------------------------------
    # Exhaustive search
    # ------------------------------------------------------------------

    def search(
        self,
        model_data: Dict[str, Dict[str, ModelQuestionData]],
        models: List[str],
        k_min: int = 2,
        k_max: int = 5,
    ) -> Dict[int, ComboResult]:
        """
        Exhaustive search over model combinations of size k_min..k_max.

        For each combination size k, evaluates every C(n, k) combination
        and returns the one with the highest score O(S).  Ties are broken
        by union_acc.

        This is a direct port of the main loop in ``offline_confacc_select.py``.

        Args:
            model_data: model_name -> {question_text -> ModelQuestionData}.
            models: Full list of available model names.
            k_min: Minimum combination size (clamped to >= 2).
            k_max: Maximum combination size (clamped to <= len(models)).

        Returns:
            Mapping of k -> ComboResult (the best combo for that size).
        """
        # Normalise range
        k_min = max(2, k_min)
        k_max = min(len(models), k_max)
        if k_min > k_max:
            k_min, k_max = k_max, k_min

        best_by_k: Dict[int, ComboResult] = {}

        for k in range(k_min, k_max + 1):
            best_rec = ComboResult(
                combo=[],
                score=0.0,
                union_acc=0.0,
                union_correct=0,
                union_total=0,
                contradiction_count=0,
            )

            for combo_tuple in itertools.combinations(models, k):
                combo_list = list(combo_tuple)
                cs = self.compute_score(combo_list, model_data)

                # Is this combo better?  Primary: score, secondary: union_acc
                is_better = (cs.score > best_rec.score) or (
                    cs.score == best_rec.score
                    and cs.union_acc > best_rec.union_acc
                )

                if is_better:
                    best_rec = ComboResult(
                        combo=combo_list,
                        score=cs.score,
                        union_acc=cs.union_acc,
                        union_correct=cs.union_correct,
                        union_total=cs.union_total,
                        contradiction_count=cs.contradiction_count,
                    )

            best_by_k[k] = best_rec
            logger.info(
                "K=%d: score=%.4f union_acc=%.4f | %s",
                k,
                best_rec.score,
                best_rec.union_acc,
                ", ".join(best_rec.combo),
            )

        return best_by_k

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def results_to_dict(
        self, best_by_k: Dict[int, ComboResult]
    ) -> Dict[str, Any]:
        """Convert search results to a JSON-serialisable dict."""
        return {
            "lambda_c": self.lambda_c,
            "robust_level": self.robust_level,
            "best_by_k": {
                str(k): {
                    "combo": r.combo,
                    "score": r.score,
                    "union_acc": r.union_acc,
                    "union_correct": r.union_correct,
                    "union_total": r.union_total,
                    "contradiction_count": r.contradiction_count,
                }
                for k, r in best_by_k.items()
            },
        }

    @staticmethod
    def save_results(
        results_dict: Dict[str, Any], path: str
    ) -> None:
        """Write search results to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
        logger.info("Saved selector results to %s", path)
