"""MATH-500 benchmark task."""

import json
import random
from typing import Any, Dict, List

from slm_mux.benchmarks.base import BenchmarkItem, BenchmarkTask
from slm_mux.extractors.math import MathExtractor, normalize_answer

# ---------------------------------------------------------------------------
# Prompts (ported from existing collection scripts)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Solve the following problem step by step. "
    "Review what you have done and make sure you have not made any mistakes. "
    "Be careful with intervals and plus or minus signs. Those parts are very easy to make mistakes. "
    "Provide the final answer enclosed in LaTeX \\boxed{...}."
)

USER_TEMPLATE = "Problem:\n{problem}\n"

FORMAT_PROMPT = "\nPlease provide your final answer in the form: \\boxed{{...}}"


class MATH500Task(BenchmarkTask):
    """MATH-500 evaluation task.

    Expects a JSON file where each element has ``"problem"`` and ``"answer"``
    fields.
    """

    def __init__(self) -> None:
        self._extractor = MathExtractor()

    # -- dataset --------------------------------------------------------

    def load_dataset(
        self, path: str, sample_size: int = -1, seed: int = 42
    ) -> List[BenchmarkItem]:
        with open(path, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)

        items = [
            BenchmarkItem(
                question=entry["problem"],
                reference_answer=entry["answer"],
                metadata={k: v for k, v in entry.items() if k not in ("problem", "answer")},
            )
            for entry in raw
        ]

        if sample_size > 0 and sample_size < len(items):
            rng = random.Random(seed)
            items = rng.sample(items, sample_size)

        return items

    # -- prompts --------------------------------------------------------

    def build_messages(self, item: BenchmarkItem) -> List[Dict[str, str]]:
        user_content = USER_TEMPLATE.format(problem=item.question) + FORMAT_PROMPT
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # -- extraction & checking -----------------------------------------

    def extract_answer(self, response: str, item: BenchmarkItem) -> str:
        return self._extractor.extract(response)

    def check_correct(self, extracted: str, reference: str, **kwargs) -> bool:
        norm_extracted = normalize_answer(extracted) if extracted else ""
        norm_reference = normalize_answer(reference) if reference else ""
        return norm_extracted == norm_reference

    # -- properties -----------------------------------------------------

    @property
    def name(self) -> str:
        return "math500"

    @property
    def question_field(self) -> str:
        return "problem"

    @property
    def reference_field(self) -> str:
        return "answer"
