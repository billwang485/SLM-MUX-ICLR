import logging
import re
from typing import Any, Dict, List, Optional

from .base import ConfidenceEvaluator, ConfidenceResult

logger = logging.getLogger(__name__)

# The follow-up prompt sent to the model to elicit a confidence rating.
_CONFIDENCE_PROMPT_TEMPLATE = (
    "You just answered a question. Your answer was:\n\n"
    "{answer}\n\n"
    "How confident are you in your answer? "
    "Rate your confidence from 0 to 100, where 0 means completely unsure "
    "and 100 means absolutely certain. "
    "Respond with ONLY a single integer number, nothing else."
)


def _parse_confidence_score(text: str) -> Optional[float]:
    """
    Parse a numeric confidence value (0-100) from model output.

    Handles common formats:
        - Plain number: "85"
        - With percent sign: "85%"
        - Embedded in short text: "My confidence is 85."
        - Decimal: "85.5"

    Returns:
        Float in [0.0, 1.0] or None if parsing fails.
    """
    text = text.strip()

    # Try to find a number in the response
    patterns = [
        r"^(\d+(?:\.\d+)?)\s*%?\s*$",  # Plain number, optionally with %
        r"(\d+(?:\.\d+)?)\s*(?:out of 100|/100)",  # "85 out of 100"
        r"(?:confidence|score|rating)[:\s]*(\d+(?:\.\d+)?)",  # "confidence: 85"
        r"(\d+(?:\.\d+)?)",  # Fallback: first number found
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Clamp to [0, 100] then normalize to [0, 1]
            value = max(0.0, min(100.0, value))
            return value / 100.0

    return None


class VerbalizedConfidence(ConfidenceEvaluator):
    """
    Verbalized confidence: Ask the model to self-report its confidence.

    After generating the answer, sends a follow-up prompt asking the model
    to rate its confidence from 0 to 100. Parses the numeric response.

    This requires a provider to make the follow-up API call.
    """

    def __init__(self, provider: Any = None, model_name: str = ""):
        """
        Args:
            provider: An object with a `chat_completion(messages, model, ...)` method
                that can call the target model. This is typically an OpenAI-compatible
                API client wrapper.
                TODO: Define a formal Provider protocol/interface.
            model_name: The model identifier to use for the follow-up call.
        """
        self._provider = provider
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "verbalized"

    @property
    def requires_external_model(self) -> bool:
        return True

    def evaluate(
        self,
        responses: List[str],
        extracted_answers: List[str],
        question: Optional[str] = None,
        logprobs: Optional[List[Dict]] = None,
    ) -> ConfidenceResult:
        """
        Evaluate confidence by asking the model to self-rate.

        Takes the first response and its extracted answer, builds a follow-up
        message asking for a 0-100 confidence rating, calls the provider,
        and parses the numeric value.

        Args:
            responses: Raw model outputs. The first response is used.
            extracted_answers: Post-extraction answers. The first is used.
            question: Original question text (used to build context).
            logprobs: Not used by this method.

        Returns:
            ConfidenceResult with score normalized to [0, 1].
        """
        if not responses:
            return ConfidenceResult(
                score=0.0,
                selected_answer="",
                metadata={"method": "verbalized", "error": "no responses provided"},
            )

        selected_answer = extracted_answers[0] if extracted_answers else ""
        first_response = responses[0]

        # Build the follow-up prompt
        confidence_prompt = _CONFIDENCE_PROMPT_TEMPLATE.format(answer=first_response)

        # Build message history for the follow-up call
        messages = []
        if question:
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": first_response})
        messages.append({"role": "user", "content": confidence_prompt})

        # Call the provider to get the confidence rating
        raw_score_text = self._call_provider(messages)

        if raw_score_text is None:
            return ConfidenceResult(
                score=0.0,
                selected_answer=selected_answer,
                metadata={
                    "method": "verbalized",
                    "error": "provider call failed or not configured",
                },
            )

        parsed_score = _parse_confidence_score(raw_score_text)

        if parsed_score is None:
            logger.warning(
                "Failed to parse confidence score from model output: %r",
                raw_score_text,
            )
            return ConfidenceResult(
                score=0.0,
                selected_answer=selected_answer,
                metadata={
                    "method": "verbalized",
                    "error": "failed to parse score",
                    "raw_output": raw_score_text,
                },
            )

        return ConfidenceResult(
            score=parsed_score,
            selected_answer=selected_answer,
            metadata={
                "method": "verbalized",
                "raw_output": raw_score_text,
                "parsed_score_0_100": parsed_score * 100,
            },
        )

    def _call_provider(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Call the provider to get the model's verbalized confidence.

        Returns:
            The raw text output from the model, or None if the call fails.
        """
        if self._provider is None:
            logger.warning(
                "VerbalizedConfidence: no provider configured. "
                "Set a provider via __init__ to enable follow-up calls."
            )
            return None

        try:
            # TODO: Adapt to the actual provider interface once it is finalized.
            # Expected interface: provider.chat_completion(messages, model, **kwargs)
            # The provider should return a response object with a .text or
            # .choices[0].message.content attribute.
            response = self._provider.chat_completion(
                messages=messages,
                model=self._model_name,
                max_tokens=16,
                temperature=0.0,
            )

            # Handle different response formats
            if isinstance(response, str):
                return response
            if hasattr(response, "text"):
                return response.text
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content
            return str(response)

        except Exception as e:
            logger.error("VerbalizedConfidence provider call failed: %s", e)
            return None
