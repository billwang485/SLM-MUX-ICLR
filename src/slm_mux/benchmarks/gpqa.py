"""GPQA benchmark task."""

import json
import random
from typing import Any, Dict, List

from slm_mux.benchmarks.base import BenchmarkItem, BenchmarkTask
from slm_mux.extractors.mcqa import MCQAExtractor

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a PhD student in science. "
    "Reason step by step through the following question and provide the best answer. "
    "Review what you have done and make sure you have not made any mistakes. "
    "Give your single-letter choice as '##Answer: X##'."
)


class GPQATask(BenchmarkTask):
    """GPQA evaluation task.

    Expects a JSON file where each element has ``"question"``, ``"choices"``,
    and ``"answer"`` fields.
    """

    def __init__(self) -> None:
        self._extractor = MCQAExtractor()

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
                choices=entry.get("choices"),
                metadata={
                    k: v
                    for k, v in entry.items()
                    if k not in ("question", "answer", "choices")
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
        choices = item.choices or ["A", "B", "C", "D"]
        user_msg = "Question:\n" + item.question + "\n\nChoices:\n"
        for choice in choices:
            user_msg += choice + "\n"
        user_msg += "\nPlease provide your final single-letter answer in the format: ##Answer: X##."

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    # -- extraction & checking -----------------------------------------

    def extract_answer(self, response: str, item: BenchmarkItem) -> str:
        choices = item.choices or ["A", "B", "C", "D"]
        return self._extractor.extract(response, valid_choices=choices)

    def check_correct(self, extracted: str, reference: str, **kwargs) -> bool:
        return extracted.strip().upper() == reference.strip().upper()

    # -- properties -----------------------------------------------------

    @property
    def name(self) -> str:
        return "gpqa"

    @property
    def question_field(self) -> str:
        return "question"

    @property
    def reference_field(self) -> str:
        return "answer"
