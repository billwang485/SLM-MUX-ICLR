"""
slm_mux.engine.mux -- Live MUX Engine (Algorithm 1 from the paper).

This module implements the core SLM-MUX algorithm for live inference:
1. For each question, generate k samples from each model independently.
2. Evaluate confidence per model using a ConfidenceEvaluator.
3. Select the model with the highest confidence (tie-break if needed).
4. Return that model's selected answer.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from slm_mux.providers.base import ProviderBase, GenerationResult
from slm_mux.confidence.base import ConfidenceEvaluator, ConfidenceResult
from slm_mux.benchmarks.base import BenchmarkTask, BenchmarkItem
from slm_mux.engine.tie_breaker import (
    TieBreaker,
    ValidationAccuracyTieBreaker,
    ModelConfidence,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Result data classes
# -----------------------------------------------------------------------

@dataclass
class MUXItemResult:
    """Result of processing a single benchmark item through the MUX algorithm."""

    question: str
    reference_answer: str
    final_answer: str
    is_correct: bool
    selected_model: str
    confidence_score: float
    all_agree: bool
    model_results: Dict[str, Any] = field(default_factory=dict)
    # model_results maps model_name -> {samples, vote_counts, best_answer, confidence, token_usage}


@dataclass
class MUXReport:
    """Aggregate report after running MUX on a full benchmark set."""

    accuracy: float
    correct_count: int
    total_count: int
    results: List[MUXItemResult]
    total_token_usage: Dict[str, int] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


# -----------------------------------------------------------------------
# Core engine
# -----------------------------------------------------------------------

class MUXEngine:
    """
    Live SLM-MUX: orchestrate multiple models with confidence-based selection.

    Algorithm (Algorithm 1 from the paper):
      1. For each question, generate k samples from each model independently.
      2. Evaluate confidence per model using the ConfidenceEvaluator
         (e.g. majority-vote consistency).
      3. Select the model with the highest confidence (tie-break if needed).
      4. Return that model's selected answer.
    """

    def __init__(
        self,
        providers: Dict[str, ProviderBase],  # model_name -> provider
        confidence_evaluator: ConfidenceEvaluator,
        benchmark: BenchmarkTask,
        models: List[str],
        samples_per_model: int = 5,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        tie_breaker: Optional[TieBreaker] = None,
        model_accuracies: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            providers: Mapping from model name to its ProviderBase instance.
            confidence_evaluator: Strategy to evaluate confidence per model
                (e.g. ConsistencyConfidence).
            benchmark: Benchmark task providing build_messages, extract_answer,
                and check_correct.
            models: Ordered list of model names to query. Order matters for
                tie-breaking.
            samples_per_model: Number of independent samples (k) to draw from
                each model per question.
            temperature: Sampling temperature for generation.
            max_tokens: Maximum tokens per generation call (None for default).
            tie_breaker: Strategy when multiple models share the top confidence.
                Defaults to ValidationAccuracyTieBreaker.
            model_accuracies: Optional model_name -> validation accuracy mapping,
                used by ValidationAccuracyTieBreaker.
        """
        self.providers = providers
        self.confidence_evaluator = confidence_evaluator
        self.benchmark = benchmark
        self.models = models
        self.samples_per_model = samples_per_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tie_breaker = tie_breaker or ValidationAccuracyTieBreaker()
        self.model_accuracies = model_accuracies or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_samples(
        self, model_name: str, item: BenchmarkItem
    ) -> Dict[str, Any]:
        """Generate k samples from one model for one question."""
        provider = self.providers[model_name]
        messages = self.benchmark.build_messages(item)

        samples: List[Dict[str, Any]] = []
        extracted_answers: List[str] = []
        total_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for i in range(self.samples_per_model):
            result: GenerationResult = provider.generate(
                model_name=model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            extracted = self.benchmark.extract_answer(result.content, item)
            samples.append(
                {
                    "round": i + 1,
                    "model_response": result.content,
                    "extracted_answer": extracted,
                    "token_usage": result.token_usage,
                }
            )
            extracted_answers.append(extracted)
            for k in total_usage:
                total_usage[k] += result.token_usage.get(k, 0)

        return {
            "model_name": model_name,
            "samples": samples,
            "extracted_answers": extracted_answers,
            "raw_responses": [s["model_response"] for s in samples],
            "token_usage": total_usage,
        }

    # ------------------------------------------------------------------
    # Per-item processing
    # ------------------------------------------------------------------

    def process_item(self, item: BenchmarkItem) -> MUXItemResult:
        """
        Process a single benchmark item through the full MUX algorithm.

        Steps:
          1. Generate k samples from each model.
          2. Evaluate per-model confidence.
          3. Find the highest confidence.
          4. Tie-break among models sharing the max confidence.
          5. Check correctness and record result.
        """
        model_results: Dict[str, Dict[str, Any]] = {}
        model_confidences: List[ModelConfidence] = []

        # Step 1: Generate samples from each model
        for model_name in self.models:
            try:
                result = self._generate_samples(model_name, item)
                model_results[model_name] = result
            except Exception as e:
                logger.error(f"Failed to generate from {model_name}: {e}")
                continue

        if not model_results:
            return MUXItemResult(
                question=item.question,
                reference_answer=item.reference_answer,
                final_answer="",
                is_correct=False,
                selected_model="none",
                confidence_score=0.0,
                all_agree=False,
            )

        # Step 2: Evaluate confidence per model
        for model_name, result in model_results.items():
            conf: ConfidenceResult = self.confidence_evaluator.evaluate(
                responses=result["raw_responses"],
                extracted_answers=result["extracted_answers"],
                question=item.question,
            )
            model_confidences.append(
                ModelConfidence(
                    model_name=model_name,
                    confidence_score=conf.score,
                    selected_answer=conf.selected_answer,
                    metadata=conf.metadata,
                )
            )
            model_results[model_name]["confidence"] = conf

        # Step 3: Find maximum confidence score
        max_conf = max(mc.confidence_score for mc in model_confidences)
        top_candidates = [
            mc
            for mc in model_confidences
            if abs(mc.confidence_score - max_conf) < 1e-9
        ]

        # Maintain model_list order for deterministic tie-breaking
        ordered_candidates: List[ModelConfidence] = []
        for m in self.models:
            for c in top_candidates:
                if c.model_name == m:
                    ordered_candidates.append(c)

        # Step 4: Tie-break
        winner = self.tie_breaker.break_tie(
            ordered_candidates, self.model_accuracies
        )

        # Step 5: Check correctness
        is_correct = self.benchmark.check_correct(
            winner.selected_answer, item.reference_answer
        )

        # Check if all models agree on the same answer
        all_answers = [mc.selected_answer for mc in model_confidences]
        all_agree = len(set(all_answers)) == 1 and all_answers[0] != ""

        return MUXItemResult(
            question=item.question,
            reference_answer=item.reference_answer,
            final_answer=winner.selected_answer,
            is_correct=is_correct,
            selected_model=winner.model_name,
            confidence_score=winner.confidence_score,
            all_agree=all_agree,
            model_results={
                mn: {
                    "samples": mr["samples"],
                    "confidence": mr.get("confidence"),
                    "token_usage": mr["token_usage"],
                }
                for mn, mr in model_results.items()
            },
        )

    # ------------------------------------------------------------------
    # Full benchmark run
    # ------------------------------------------------------------------

    def run(
        self,
        items: List[BenchmarkItem],
        num_workers: int = 10,
        progress: bool = True,
    ) -> MUXReport:
        """
        Run MUX on all items with thread-pool parallelism.

        Args:
            items: List of benchmark items to process.
            num_workers: Number of concurrent threads.
            progress: Whether to display a tqdm progress bar.

        Returns:
            MUXReport with aggregate accuracy and per-item results.
        """
        results: List[MUXItemResult] = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(self.process_item, item): item for item in items
            }

            pbar = None
            if progress:
                try:
                    from tqdm import tqdm

                    pbar = tqdm(total=len(futures), desc="SLM-MUX", unit="q")
                except ImportError:
                    logger.warning("tqdm not installed; disabling progress bar")

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item: {e}")

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

        correct = sum(1 for r in results if r.is_correct)
        total = len(results)

        # Aggregate token usage across all models and items
        total_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for r in results:
            for mr in r.model_results.values():
                for k in total_usage:
                    total_usage[k] += mr.get("token_usage", {}).get(k, 0)

        return MUXReport(
            accuracy=correct / total if total > 0 else 0.0,
            correct_count=correct,
            total_count=total,
            results=results,
            total_token_usage=total_usage,
        )
