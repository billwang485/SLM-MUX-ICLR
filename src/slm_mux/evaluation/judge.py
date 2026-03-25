"""LLM-as-judge for answer equivalence verification.

Ported from single_model_inference/collect_math.py verify_with_llm_gpt4o().
"""

import logging
import time
from typing import Any, Dict, Optional

import requests

from slm_mux.providers._secrets import load_api_key

logger = logging.getLogger(__name__)

EQUAL_VERIFY_PROMPT = """You are a mathematics teacher grading a student's answer on a test.

The standard (correct) answer is: {reference}

The student wrote: {extracted}

As a fair but rigorous math teacher:
- Would you mark the student's answer as CORRECT?
- The student's notation may differ from yours (e.g., using 3 vs 3^\\circ for angles, or 1/sqrt(3) vs sqrt(3)/3)
- If the student's answer is mathematically equivalent to the standard answer, it should be marked correct
- Minor notational differences are acceptable as long as the mathematical meaning is preserved
- The student's answer must represent exactly the same value or function as the standard answer

Respond ONLY with "Yes" (I would mark it correct) or "No" (I would mark it incorrect).
Answer:""".strip()


class LLMJudge:
    """Use an LLM to judge answer equivalence."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_retries = max_retries

    def judge(self, reference: str, extracted: str) -> Dict[str, Any]:
        """
        Judge if extracted answer is equivalent to reference.

        Returns:
            {"decision": "Yes"|"No"|"", "raw": full_response_text}
        """
        api_key = load_api_key(self.api_key_env, [self.api_key_env, "openai_api_key"])
        if not api_key:
            logger.warning("No API key for LLM judge; returning empty decision.")
            return {"decision": "", "raw": ""}

        prompt = EQUAL_VERIFY_PROMPT.format(reference=reference, extracted=extracted)
        endpoint = "https://api.openai.com/v1/chat/completions"

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    endpoint,
                    json={
                        "model": self.model,
                        "temperature": self.temperature,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=(10, 60),
                )
                data = resp.json()
                if "error" in data:
                    logger.error(f"Judge API error: {data['error']}")
                    time.sleep(2 ** attempt)
                    continue

                text = data["choices"][0]["message"]["content"].strip()
                decision = text.strip().strip('"').split()[0] if text else ""
                return {"decision": decision, "raw": text}
            except Exception as e:
                logger.error(f"Judge error (attempt {attempt + 1}): {e}")
                time.sleep(2 ** attempt)

        return {"decision": "", "raw": ""}

    def is_correct(self, reference: str, extracted: str) -> Optional[bool]:
        """Convenience: returns True/False/None based on judge decision."""
        result = self.judge(reference, extracted)
        decision = result["decision"].lower()
        if decision.startswith("yes"):
            return True
        elif decision.startswith("no"):
            return False
        return None
