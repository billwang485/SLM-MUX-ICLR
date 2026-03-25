"""Abstract base class for answer extractors."""

from abc import ABC, abstractmethod


class AnswerExtractor(ABC):
    """Base class for extracting structured answers from free-form model responses."""

    @abstractmethod
    def extract(self, response: str, **kwargs) -> str:
        """Extract answer from model response.

        Args:
            response: Raw text output from the model.
            **kwargs: Extractor-specific options.

        Returns:
            Extracted answer string, or empty string if extraction fails.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this extractor."""
        ...
