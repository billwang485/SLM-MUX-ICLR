"""GSM8K benchmark task."""

import json
import random
from typing import Any, Dict, List

from slm_mux.benchmarks.base import BenchmarkItem, BenchmarkTask
from slm_mux.extractors.gsm import GSMExtractor, normalize_answer_gsm

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Reason step by step to solve the problem. "
    "After your reasoning, output the final answer on a new line prefixed with '####'. "
    "Example: #### 42"
)


class GSM8KTask(BenchmarkTask):
    """GSM8K evaluation task.

    Expects a JSON file where each element has ``"question"`` and ``"answer"``
    fields.
    """

    def __init__(self) -> None:
        self._extractor = GSMExtractor()

    # -- dataset --------------------------------------------------------

    def load_dataset(
        self, path: str, sample_size: int = -1, seed: int = 42
    ) -> List[BenchmarkItem]:
        with open(path, "r", encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)

        items = [
            BenchmarkItem(
                question=entry["question"],
                reference_answer=entry["answer"],
                metadata={
                    k: v for k, v in entry.items() if k not in ("question", "answer")
                },
            )
            for entry in raw
        ]

        if sample_size > 0 and sample_size < len(items):
            rng = random.Random(seed)
            items = rng.sample(items, sample_size)

        return items

    # -- prompts --------------------------------------------------------

    def build_messages(self, item: BenchmarkItem) -> List[Dict[str, str]]:
        user_msg = "Problem:\n" + item.question
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    # -- extraction & checking -----------------------------------------

    def extract_answer(self, response: str, item: BenchmarkItem) -> str:
        return self._extractor.extract(response)

    def check_correct(self, extracted: str, reference: str, **kwargs) -> bool:
        norm_extracted = normalize_answer_gsm(extracted) if extracted else ""
        norm_reference = normalize_answer_gsm(reference) if reference else ""
        return norm_extracted == norm_reference

    # -- properties -----------------------------------------------------

    @property
    def name(self) -> str:
        return "gsm8k"

    @property
    def question_field(self) -> str:
        return "question"

    @property
    def reference_field(self) -> str:
        return "answer"
