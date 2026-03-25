"""Code block extractor for HumanEval-style benchmarks.

Extracts the content of the first ```python ... ``` fenced code block from a
model response.
"""

import re

from slm_mux.extractors.base import AnswerExtractor


def extract_code_block(text: str, language: str = "python") -> str:
    """Extract content from the first fenced code block matching *language*.

    Tries language-specific fences first (e.g. ``python``), then falls back
    to bare triple-backtick fences.

    Args:
        text: Raw model response.
        language: Expected language tag (default ``"python"``).

    Returns:
        Code string with leading/trailing whitespace stripped, or empty string
        if no code block is found.
    """
    # Try language-specific fence first
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    m = re.search(pattern, text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fall back to bare triple-backtick fence
    m = re.search(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    return ""


class CodeExtractor(AnswerExtractor):
    """Extractor for fenced code blocks (HumanEval / code-generation tasks).

    Args:
        language: Expected language tag inside the code fence (default
            ``"python"``).
    """

    def __init__(self, language: str = "python") -> None:
        self._language = language

    def extract(self, response: str, **kwargs) -> str:
        """Extract the first fenced code block from the response.

        Keyword Args:
            language: Override the default language for this call.
        """
        lang = kwargs.get("language", self._language)
        return extract_code_block(response, language=lang)

    @property
    def name(self) -> str:
        return "code"
