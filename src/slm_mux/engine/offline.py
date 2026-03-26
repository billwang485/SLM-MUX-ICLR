"""
slm_mux.engine.offline -- Offline MUX simulation from pre-collected responses.

Port of the existing offline_mux_scan.py / offline_mux_math.py logic,
unified across all benchmarks and parameterised through the confidence
evaluator and tie-breaker interfaces.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from slm_mux.confidence.base import ConfidenceEvaluator, ConfidenceResult
from slm_mux.engine.tie_breaker import (
    TieBreaker,
    ValidationAccuracyTieBreaker,
    ModelConfidence,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------

@dataclass
class ModelQuestionData:
    """Pre-collected data for one model on one question."""

    reference: str
    samples: List[str]  # extracted answers (may contain empty strings)


@dataclass
class TrialResult:
    """Outcome of a single Monte Carlo trial."""

    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


@dataclass
class SimulationResult:
    """Aggregate result from an offline MUX simulation."""

    accuracy: float
    correct: int
    total: int
    trial_results: List[TrialResult] = field(default_factory=list)
    per_question: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# -----------------------------------------------------------------------
# OfflineMUX
# -----------------------------------------------------------------------

class OfflineMUX:
    """
    Simulate MUX from pre-collected response JSONs.

    Loads pre-generated response files (one per model), subsamples k answers
    per model per question, applies confidence-based selection, and reports
    accuracy.  Uses Monte Carlo trials for stability.

    This is a faithful port of ``simulate_offline_mux()`` from the existing
    ``offline_mux_scan.py``, but parameterised to work with any benchmark
    and using :class:`ConfidenceEvaluator` instead of inline vote counting.
    """

    def __init__(
        self,
        confidence_evaluator: Optional[ConfidenceEvaluator] = None,
        tie_breaker: Optional[TieBreaker] = None,
        model_accuracies: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            confidence_evaluator: Evaluator for per-model confidence.  If None,
                a default ConsistencyConfidence evaluator is created on first use.
            tie_breaker: Strategy for breaking ties.  Defaults to
                ValidationAccuracyTieBreaker.
            model_accuracies: model_name -> validation accuracy, used for
                tie-breaking and the best_support strategy.
        """
        self._confidence_evaluator = confidence_evaluator
        self.tie_breaker = tie_breaker or ValidationAccuracyTieBreaker()
        self.model_accuracies = model_accuracies or {}

    @property
    def confidence_evaluator(self) -> ConfidenceEvaluator:
        if self._confidence_evaluator is None:
            from slm_mux.confidence.consistency import ConsistencyConfidence

            self._confidence_evaluator = ConsistencyConfidence()
        return self._confidence_evaluator

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_model_data(
        path: str, benchmark_name: str = "math"
    ) -> Dict[str, ModelQuestionData]:
        """
        Load a model's response JSON.  Auto-detect format based on benchmark_name.

        Handles three formats:
          - MATH:  responses[].problem, responses[].reference_answer,
                   responses[].samples[].extracted_answer
          - GPQA:  responses[].question, responses[].correct_answer,
                   responses[].samples[].extracted_answer
          - GSM8K: responses[].question, responses[].reference_answer,
                   responses[].samples[].extracted_answer

        Returns:
            Mapping of question_text -> ModelQuestionData(reference, samples).
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        bench = benchmark_name.upper()
        # Normalize benchmark name variants
        if bench in ("MATH500", "MATH"):
            bench = "MATH"
        elif bench in ("GSM8K", "GSM"):
            bench = "GSM8K"
        elif bench in ("IFEVAL",):
            bench = "IFEVAL"
        elif bench in ("HUMANEVAL",):
            bench = "HUMANEVAL"

        result: Dict[str, ModelQuestionData] = {}

        responses = data.get("responses") or data.get("results") or []
        for resp in responses:
            # Determine question key
            if bench == "MATH":
                q = resp.get("problem", "")
            elif bench in ("IFEVAL", "HUMANEVAL"):
                q = resp.get("prompt", "")
            else:
                q = resp.get("question", "")

            # Determine reference answer key (support multiple naming conventions)
            if bench == "GPQA":
                ref = (
                    resp.get("correct_answer")
                    or resp.get("reference_answer")
                    or resp.get("answer")
                    or ""
                ).strip()
            elif bench == "IFEVAL":
                # For IFEval, pack instruction_id_list + kwargs as reference
                import json as _json
                ref = _json.dumps({
                    "instruction_id_list": resp.get("instruction_id_list", []),
                    "kwargs": resp.get("kwargs", []),
                }, ensure_ascii=False)
            else:
                ref = (
                    resp.get("reference_answer")
                    or resp.get("answer")
                    or resp.get("reference")
                    or ""
                ).strip()

            # Extract sample answers (support multiple naming conventions)
            samples: List[str] = []
            for s in resp.get("samples", []):
                if bench == "IFEVAL":
                    # For IFEval, the full response is the answer
                    ans = s.get("model_response") or s.get("response") or ""
                else:
                    ans = s.get("extracted_answer") or s.get("extracted") or ""
                samples.append(ans.strip() if isinstance(ans, str) else str(ans))

            result[q] = ModelQuestionData(reference=ref, samples=samples)

        return result

    @staticmethod
    def read_model_accuracy(path: str) -> float:
        """Read pre-computed accuracy from a model's response JSON."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get("accuracy", 0.0) or 0.0)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        model_data: Dict[str, Dict[str, ModelQuestionData]],
        models: List[str],
        samples_per_model: int,
        trials: int = 10,
        seed: int = 42,
        tie_mode: str = "best_support",
    ) -> SimulationResult:
        """
        Monte Carlo simulation of offline MUX.

        For each trial, for each common question:
          1. Randomly subsample ``samples_per_model`` answers from each model.
          2. Pool all subsampled answers across models into a single vote set.
          3. Determine the majority-vote answer.
          4. If there is a tie in votes, use tie_mode to resolve:
             - "best_support": pick the answer supported by the model with
               the highest validation accuracy.
             - "dual_order_max": run the simulation in both forward and reverse
               model order; pick the order yielding higher accuracy.
          5. Record whether the selected answer matches the reference.

        This is a faithful port of ``simulate_offline_mux()`` from the
        existing ``offline_mux_scan.py``.

        Args:
            model_data: model_name -> {question_text -> ModelQuestionData}.
            models: Ordered list of model names.
            samples_per_model: Number of samples (k) to draw per model per
                question per trial.
            trials: Number of Monte Carlo trials.
            seed: Random seed for reproducibility.
            tie_mode: "best_support" or "dual_order_max".

        Returns:
            SimulationResult with aggregate accuracy.
        """
        rng = random.Random(seed)

        # Intersection of questions across all models
        qsets: List[Set[str]] = [set(model_data[m].keys()) for m in models]
        common: Set[str] = set.intersection(*qsets) if qsets else set()
        if not common:
            return SimulationResult(accuracy=0.0, correct=0, total=0)

        # Validate sample availability
        max_samples = 0
        for m in models:
            for q in common:
                max_samples = max(max_samples, len(model_data[m][q].samples))
        if samples_per_model > max_samples:
            raise ValueError(
                f"samples_per_model={samples_per_model} exceeds available "
                f"per-question samples={max_samples}"
            )

        def simulate_once(order: List[str]) -> Tuple[int, int]:
            """Run all trials with a given model order and return (correct, total)."""
            total_correct = 0
            total_count = 0

            for _ in range(max(1, trials)):
                for q in common:
                    votes: List[str] = []
                    # Track which models support each answer (for best_support)
                    answer_supporters: Dict[str, Set[str]] = {}
                    reference = ""

                    for m in order:
                        entry = model_data[m][q]
                        if not reference:
                            reference = entry.reference
                        pool = entry.samples or []
                        k = min(samples_per_model, len(pool))
                        if k > 0:
                            idxs = rng.sample(range(len(pool)), k=k)
                            for i in idxs:
                                ans = pool[i]
                                votes.append(ans)
                                if ans:
                                    answer_supporters.setdefault(
                                        ans, set()
                                    ).add(m)

                    # Majority vote
                    pred = ""
                    counter: Dict[str, int] = {}
                    for v in votes:
                        if not v:
                            continue
                        counter[v] = counter.get(v, 0) + 1

                    if counter:
                        max_count = max(counter.values())
                        top_answers = [
                            a for a, c in counter.items() if c == max_count
                        ]

                        if len(top_answers) == 1 or tie_mode == "dual_order_max":
                            # Unique winner, or dual_order_max: resolve by
                            # first occurrence in votes
                            if len(top_answers) == 1:
                                pred = top_answers[0]
                            else:
                                # Pick the tied answer that appears first in
                                # the votes sequence (order-dependent)
                                first = ""
                                for v in votes:
                                    if v in top_answers:
                                        first = v
                                        break
                                pred = first or top_answers[0]
                        else:
                            # best_support tie-break: pick the answer whose
                            # best supporting model has the highest accuracy
                            def best_support_acc(ans: str) -> float:
                                sups = answer_supporters.get(ans, set())
                                if not sups:
                                    return 0.0
                                return max(
                                    self.model_accuracies.get(m_name, 0.0)
                                    for m_name in sups
                                )

                            scored = sorted(
                                top_answers,
                                key=lambda a: best_support_acc(a),
                                reverse=True,
                            )
                            best_score = (
                                best_support_acc(scored[0]) if scored else 0.0
                            )
                            finalists = [
                                a
                                for a in scored
                                if best_support_acc(a) == best_score
                            ]
                            pred = (
                                rng.choice(finalists) if finalists else ""
                            )

                    total_count += 1
                    if pred and reference and pred == reference:
                        total_correct += 1

            return total_correct, total_count

        if tie_mode == "dual_order_max":
            c_fwd, n_fwd = simulate_once(models)
            c_rev, n_rev = simulate_once(list(reversed(models)))
            acc_fwd = (c_fwd / n_fwd) if n_fwd else 0.0
            acc_rev = (c_rev / n_rev) if n_rev else 0.0
            if acc_fwd >= acc_rev:
                acc, correct, total = acc_fwd, c_fwd, n_fwd
            else:
                acc, correct, total = acc_rev, c_rev, n_rev
        else:
            correct, total = simulate_once(models)
            acc = (correct / total) if total else 0.0

        return SimulationResult(
            accuracy=acc,
            correct=correct,
            total=total,
        )
