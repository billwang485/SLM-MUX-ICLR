"""Benchmark task definitions for SLM-MUX evaluation."""

from slm_mux.benchmarks.base import BenchmarkTask, BenchmarkItem
from slm_mux.benchmarks.math500 import MATH500Task
from slm_mux.benchmarks.gpqa import GPQATask
from slm_mux.benchmarks.gsm8k import GSM8KTask
from slm_mux.benchmarks.ifeval import IFEvalTask
from slm_mux.benchmarks.humaneval import HumanEvalTask

__all__ = [
    "BenchmarkTask",
    "BenchmarkItem",
    "MATH500Task",
    "GPQATask",
    "GSM8KTask",
    "IFEvalTask",
    "HumanEvalTask",
]
