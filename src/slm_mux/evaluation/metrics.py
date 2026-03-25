"""Evaluation metrics for SLM-MUX."""

import math
from typing import List, Optional


def accuracy(correct_count: int, total_count: int) -> float:
    """Compute accuracy."""
    return correct_count / total_count if total_count > 0 else 0.0


def accuracy_with_se(correct_count: int, total_count: int) -> tuple:
    """Compute accuracy with standard error."""
    if total_count == 0:
        return 0.0, 0.0
    acc = correct_count / total_count
    se = math.sqrt(acc * (1 - acc) / total_count)
    return acc, se


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k metric.

    Args:
        n: Total number of samples generated.
        c: Number of correct samples.
        k: k value for pass@k.

    Returns:
        Estimated pass@k probability.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def consistency_accuracy_correlation(
    consistency_scores: List[float],
    is_correct: List[bool],
) -> Optional[float]:
    """
    Compute Pearson correlation between consistency scores and correctness.

    Returns None if computation fails (e.g., zero variance).
    """
    n = len(consistency_scores)
    if n != len(is_correct) or n < 2:
        return None

    correct_float = [1.0 if c else 0.0 for c in is_correct]
    mean_s = sum(consistency_scores) / n
    mean_c = sum(correct_float) / n

    cov = sum((s - mean_s) * (c - mean_c) for s, c in zip(consistency_scores, correct_float)) / n
    var_s = sum((s - mean_s) ** 2 for s in consistency_scores) / n
    var_c = sum((c - mean_c) ** 2 for c in correct_float) / n

    denom = math.sqrt(var_s * var_c)
    if denom < 1e-12:
        return None
    return cov / denom
