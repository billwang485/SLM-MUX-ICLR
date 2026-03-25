"""Multiple-choice question answer extractor (GPQA-style).

Ported from utils/gpqa_utils.py.  Tries several common answer formats in
order of reliability before giving up.
"""

import re
from typing import List, Optional, Union

from slm_mux.extractors.base import AnswerExtractor


def extract_answer_gpqa(text: str, valid_choices: List[str]) -> str:
    """Extract a single-letter MCQA answer from *text*.

    Tries the following strategies in priority order:
      1. ``answer ... X`` pattern (most reliable)
      2. ``## X ##`` double-hash wrapper
      3. Single letter on its own line
      4. ``(X)`` parenthesised letter (last occurrence)
      5. Whitespace + letter (last occurrence)
      6. ``Final Answer: (X)``

    Args:
        text: Raw model response.
        valid_choices: List of choice strings; only the first character of
            each entry is used for validation (e.g. ``["A", "B", "C", "D"]``
            or ``["A) ...", "B) ..."]``).

    Returns:
        Uppercase single letter, or empty string if extraction fails.
    """
    valid = {c[:1].upper() for c in valid_choices}

    # 1) "Answer: X" -- most reliable
    m = re.search(r"answer[^A-Za-z0-9]*([A-D])", text, flags=re.I)
    if m:
        ans = m.group(1).upper()
        if ans in valid:
            return ans

    # 2) ## X ##
    m = re.search(r"##\s*([A-D])\s*##", text)
    if m:
        ans = m.group(1).upper()
        if ans in valid:
            return ans

    # 3) Single letter on its own line
    for line in text.splitlines():
        s = line.strip().upper()
        if s in valid and len(s) == 1:
            return s

    # 4) (X) parentheses -- take last occurrence
    for m in reversed(re.findall(r"\(([A-D])\)", text)):
        ans = m.upper()
        if ans in valid:
            return ans

    # 5) space + letter -- take last occurrence
    for m in reversed(re.findall(r"\s([A-D])", text)):
        ans = m.upper()
        if ans in valid:
            return ans

    # 6) Final Answer: (X)
    m = re.search(r"Final Answer: \(([A-Z])\)", text)
    if m:
        ans = m.group(1).upper()
        if ans in valid:
            return ans

    return ""


class MCQAExtractor(AnswerExtractor):
    """Extractor for multiple-choice questions with fixed valid choices.

    Args:
        valid_choices: The set of valid answer options (e.g.
            ``["A", "B", "C", "D"]``).
    """

    def __init__(self, valid_choices: Optional[List[str]] = None) -> None:
        self._valid_choices = valid_choices or ["A", "B", "C", "D"]

    def extract(self, response: str, **kwargs) -> str:
        """Extract a single-letter answer from the response.

        Keyword Args:
            valid_choices: Override the default valid choices for this call.
        """
        choices = kwargs.get("valid_choices", self._valid_choices)
        return extract_answer_gpqa(response, choices)

    @property
    def name(self) -> str:
        return "mcqa"
