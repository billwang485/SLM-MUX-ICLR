"""GSM8K answer extractor.

Ported from utils/gsm_utils.py.  Handles the ``#### <number>`` convention
used by GSM8K, with a fallback to the last standalone number in the response.
"""

import re

from slm_mux.extractors.base import AnswerExtractor


def extract_answer_gsm(text: str) -> str:
    """Extract the numeric answer from a GSM8K-style response.

    Looks for the last ``#### <number>`` marker.  If none is found, falls back
    to the last standalone number in the text.

    Returns:
        The extracted number as a string, or empty string on failure.
    """
    if text is None:
        return ""
    # Take the last occurrence of #### to be safe
    matches = list(re.finditer(r"####\s*([-]?\d+(?:\.\d+)?)", text))
    if matches:
        return matches[-1].group(1)
    # Fallback: last standalone number
    nums = re.findall(r"[-]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


def normalize_answer_gsm(ans: str) -> str:
    """Normalise a GSM8K answer by stripping whitespace, commas, and leading zeros."""
    if ans is None:
        return ""
    ans = ans.strip()
    if ans.startswith("####"):
        ans = ans[4:].strip()
    ans = ans.replace(",", "")
    # Remove leading zeros (but keep a single zero)
    if re.fullmatch(r"0+", ans):
        return "0"
    ans = re.sub(r"^0+", "", ans)
    return ans


class GSMExtractor(AnswerExtractor):
    """Extractor for GSM8K ``#### <number>`` answers."""

    def extract(self, response: str, **kwargs) -> str:
        """Extract and normalise a numeric answer from the response."""
        raw = extract_answer_gsm(response)
        return normalize_answer_gsm(raw)

    @property
    def name(self) -> str:
        return "gsm"
