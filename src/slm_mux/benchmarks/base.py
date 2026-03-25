"""Abstract base class and core data structures for benchmark tasks."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkItem:
    """A single evaluation item from a benchmark dataset.

    Attributes:
        question: The problem or question text.
        reference_answer: The gold/reference answer.
        choices: Optional list of answer choices (for MCQA benchmarks).
        metadata: Arbitrary extra fields from the dataset row.
    """

    question: str
    reference_answer: str
    choices: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkTask(ABC):
    """Base class for benchmark evaluation tasks.

    Subclasses must implement dataset loading, prompt construction, answer
    extraction, and correctness checking.
    """

    @abstractmethod
    def load_dataset(
        self, path: str, sample_size: int = -1, seed: int = 42
    ) -> List[BenchmarkItem]:
        """Load the benchmark dataset from *path*.

        Args:
            path: Path to the dataset file (JSON).
            sample_size: Number of items to sample.  ``-1`` means use all.
            seed: Random seed for reproducible sampling.

        Returns:
            List of ``BenchmarkItem`` instances.
        """
        ...

    @abstractmethod
    def build_messages(self, item: BenchmarkItem) -> List[Dict[str, str]]:
        """Build the chat-style message list for a single benchmark item.

        Returns:
            A list of dicts with ``"role"`` and ``"content"`` keys.
        """
        ...

    @abstractmethod
    def extract_answer(self, response: str, item: BenchmarkItem) -> str:
        """Extract a structured answer from the raw model response.

        Args:
            response: Raw text from the model.
            item: The benchmark item (may be needed for choices, etc.).

        Returns:
            Extracted answer string.
        """
        ...

    @abstractmethod
    def check_correct(self, extracted: str, reference: str, **kwargs) -> bool:
        """Check whether the extracted answer matches the reference.

        Args:
            extracted: Answer extracted from the model response.
            reference: Gold/reference answer from the dataset.

        Returns:
            ``True`` if the answer is correct.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name."""
        ...

    @property
    def question_field(self) -> str:
        """Name of the question field in the raw dataset."""
        return "question"

    @property
    def reference_field(self) -> str:
        """Name of the reference answer field in the raw dataset."""
        return "reference_answer"
