"""IFEval benchmark task -- instruction-following evaluation.

Evaluates whether models can follow specific, verifiable constraints
such as word counts, formatting requirements, keyword presence, etc.

Dataset: google/IFEval (541 prompts, 25 instruction types).
Scoring is fully deterministic -- no LLM judge required.

Reference: Zhou et al. (2023), "Instruction-Following Evaluation for
Large Language Models", arXiv:2311.07911.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from slm_mux.benchmarks.base import BenchmarkItem, BenchmarkTask


# ===================================================================
# IFEval instruction verifiers
# ===================================================================

def _count_words(text: str) -> int:
    return len(text.split())


def _count_sentences(text: str) -> int:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return len([s for s in sentences if s.strip()])


def _count_paragraphs(text: str) -> int:
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return len([p for p in paragraphs if p.strip()])


def _get_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _check_relation(actual: int, target: int, relation: str) -> bool:
    relation = relation.lower().strip()
    if relation in ("at least", "no less than"):
        return actual >= target
    elif relation in ("at most", "no more than"):
        return actual <= target
    elif relation in ("exactly",):
        return actual == target
    elif relation in ("less than",):
        return actual < target
    elif relation in ("more than",):
        return actual > target
    return actual >= target


# --- Individual verifiers ---------------------------------------------------

def _v_keywords_existence(resp: str, keywords: Optional[List[str]] = None, **kw) -> bool:
    if not keywords:
        return True
    r = resp.lower()
    return all(k.lower() in r for k in keywords)


def _v_keywords_frequency(resp: str, keyword: Optional[str] = None,
                           frequency: Optional[int] = None,
                           relation: Optional[str] = None, **kw) -> bool:
    if keyword is None or frequency is None:
        return True
    return _check_relation(resp.lower().count(keyword.lower()),
                           frequency, relation or "at least")


def _v_keywords_forbidden_words(resp: str,
                                 forbidden_words: Optional[List[str]] = None, **kw) -> bool:
    if not forbidden_words:
        return True
    r = resp.lower()
    return not any(w.lower() in r for w in forbidden_words)


def _v_keywords_letter_frequency(resp: str, letter: Optional[str] = None,
                                  let_frequency: Optional[int] = None,
                                  let_relation: Optional[str] = None, **kw) -> bool:
    if letter is None or let_frequency is None:
        return True
    return _check_relation(resp.lower().count(letter.lower()),
                           let_frequency, let_relation or "at least")


def _v_language_response_language(resp: str, language: Optional[str] = None, **kw) -> bool:
    if not language:
        return True
    try:
        from langdetect import detect
        lang_map = {
            "english": "en", "french": "fr", "spanish": "es", "german": "de",
            "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
            "chinese": "zh-cn", "japanese": "ja", "korean": "ko", "arabic": "ar",
            "hindi": "hi", "turkish": "tr", "polish": "pl", "swedish": "sv",
            "danish": "da", "norwegian": "no", "finnish": "fi", "greek": "el",
            "czech": "cs", "romanian": "ro", "hungarian": "hu", "bulgarian": "bg",
            "croatian": "hr", "slovak": "sk", "ukrainian": "uk", "vietnamese": "vi",
            "thai": "th", "indonesian": "id", "malay": "ms", "hebrew": "he",
            "persian": "fa", "urdu": "ur", "bengali": "bn", "tamil": "ta",
            "telugu": "te", "swahili": "sw",
        }
        target = lang_map.get(language.lower(), language.lower()[:2])
        detected = detect(resp)
        return detected == target or detected.startswith(target)
    except Exception:
        return True


def _v_length_number_words(resp: str, relation: Optional[str] = None,
                            num_words: Optional[int] = None, **kw) -> bool:
    if num_words is None:
        return True
    return _check_relation(_count_words(resp), num_words, relation or "at least")


def _v_length_number_sentences(resp: str, relation: Optional[str] = None,
                                num_sentences: Optional[int] = None, **kw) -> bool:
    if num_sentences is None:
        return True
    return _check_relation(_count_sentences(resp), num_sentences, relation or "at least")


def _v_length_number_paragraphs(resp: str, num_paragraphs: Optional[int] = None, **kw) -> bool:
    if num_paragraphs is None:
        return True
    return _count_paragraphs(resp) >= num_paragraphs


def _v_length_nth_paragraph_first_word(resp: str, num_paragraphs: Optional[int] = None,
                                        nth_paragraph: Optional[int] = None,
                                        first_word: Optional[str] = None, **kw) -> bool:
    if nth_paragraph is None or first_word is None:
        return True
    paras = _get_paragraphs(resp)
    if num_paragraphs is not None and len(paras) < num_paragraphs:
        return False
    idx = nth_paragraph - 1
    if idx < 0 or idx >= len(paras):
        return False
    words = paras[idx].split()
    first = words[0] if words else ""
    return first.lower().rstrip(".,;:!?") == first_word.lower().rstrip(".,;:!?")


def _v_detectable_content_number_placeholders(resp: str,
                                               num_placeholders: Optional[int] = None, **kw) -> bool:
    if num_placeholders is None:
        return True
    return len(re.findall(r'\[.*?\]', resp)) >= num_placeholders


def _v_detectable_content_postscript(resp: str,
                                      postscript_marker: Optional[str] = None, **kw) -> bool:
    marker = postscript_marker or "P.S."
    return any(p in resp for p in [marker, "P.S.", "PS:", "P.S:", "PS.", "p.s."])


def _v_detectable_format_number_bullet_lists(resp: str,
                                              num_bullets: Optional[int] = None, **kw) -> bool:
    if num_bullets is None:
        return True
    count = 0
    for line in resp.strip().split('\n'):
        s = line.strip()
        if re.match(r'^[\*\-\u2022]\s', s) or re.match(r'^\d+[\.\)]\s', s):
            count += 1
    return count >= num_bullets


def _v_detectable_format_constrained_response(resp: str, **kw) -> bool:
    return True  # Underspecified in the official code


def _v_detectable_format_number_highlighted_sections(
    resp: str, num_highlights: Optional[int] = None, **kw,
) -> bool:
    if num_highlights is None:
        return True
    bold = len(re.findall(r'\*\*[^*]+\*\*', resp))
    italic = len(re.findall(r'(?<!\*)\*[^*]+\*(?!\*)', resp))
    return (bold + italic) >= num_highlights


def _v_detectable_format_multiple_sections(
    resp: str, section_spliter: Optional[str] = None,
    num_sections: Optional[int] = None, **kw,
) -> bool:
    if num_sections is None:
        return True
    splitter = section_spliter or "SECTION"
    return len(re.findall(re.escape(splitter), resp, re.IGNORECASE)) >= num_sections


def _v_detectable_format_json_format(resp: str, **kw) -> bool:
    text = resp.strip()
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            json.loads(m.group(1).strip())
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    for sc, ec in [('{', '}'), ('[', ']')]:
        si, ei = text.find(sc), text.rfind(ec)
        if si != -1 and ei > si:
            try:
                json.loads(text[si:ei + 1])
                return True
            except (json.JSONDecodeError, ValueError):
                pass
    return False


def _v_detectable_format_title(resp: str, **kw) -> bool:
    lines = resp.strip().split('\n')
    if not lines:
        return False
    first = lines[0].strip()
    if first.startswith('#'):
        return True
    if first.startswith('**') and first.endswith('**'):
        return True
    if len(first.split()) <= 10 and first == first.upper() and first != first.lower():
        return True
    if len(first.split()) <= 10 and first == first.title():
        return True
    return False


def _v_combination_two_responses(resp: str, **kw) -> bool:
    return bool(re.search(r'\*{5,}', resp))


def _v_combination_repeat_prompt(resp: str, prompt_to_repeat: Optional[str] = None, **kw) -> bool:
    if prompt_to_repeat is None:
        return True
    return prompt_to_repeat.strip().lower() in resp.strip()[:len(prompt_to_repeat) + 50].lower()


def _v_startend_end_checker(resp: str, end_phrase: Optional[str] = None, **kw) -> bool:
    if end_phrase is None:
        return True
    return resp.strip().endswith(end_phrase.strip())


def _v_startend_quotation(resp: str, **kw) -> bool:
    text = resp.strip()
    if not text:
        return False
    return ((text.startswith('"') and text.endswith('"')) or
            (text.startswith("'") and text.endswith("'")) or
            (text.startswith('\u201c') and text.endswith('\u201d')) or
            (text.startswith('\u2018') and text.endswith('\u2019')))


def _v_change_case_english_capital(resp: str, **kw) -> bool:
    alpha = ''.join(c for c in resp if c.isalpha())
    return (not alpha) or alpha == alpha.upper()


def _v_change_case_english_lowercase(resp: str, **kw) -> bool:
    alpha = ''.join(c for c in resp if c.isalpha())
    return (not alpha) or alpha == alpha.lower()


def _v_change_case_capital_word_frequency(
    resp: str, capital_frequency: Optional[int] = None,
    capital_relation: Optional[str] = None, **kw,
) -> bool:
    if capital_frequency is None:
        return True
    words = resp.split()
    if not words:
        return True
    count = sum(1 for w in words if w and w[0].isupper())
    return _check_relation(count, capital_frequency, capital_relation or "at least")


def _v_punctuation_no_comma(resp: str, **kw) -> bool:
    return ',' not in resp


# --- Verifier registry -------------------------------------------------------

_VERIFIERS = {
    "keywords:existence": _v_keywords_existence,
    "keywords:frequency": _v_keywords_frequency,
    "keywords:forbidden_words": _v_keywords_forbidden_words,
    "keywords:letter_frequency": _v_keywords_letter_frequency,
    "language:response_language": _v_language_response_language,
    "length_constraints:number_words": _v_length_number_words,
    "length_constraints:number_sentences": _v_length_number_sentences,
    "length_constraints:number_paragraphs": _v_length_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _v_length_nth_paragraph_first_word,
    "detectable_content:number_placeholders": _v_detectable_content_number_placeholders,
    "detectable_content:postscript": _v_detectable_content_postscript,
    "detectable_format:number_bullet_lists": _v_detectable_format_number_bullet_lists,
    "detectable_format:constrained_response": _v_detectable_format_constrained_response,
    "detectable_format:number_highlighted_sections": _v_detectable_format_number_highlighted_sections,
    "detectable_format:multiple_sections": _v_detectable_format_multiple_sections,
    "detectable_format:json_format": _v_detectable_format_json_format,
    "detectable_format:title": _v_detectable_format_title,
    "combination:two_responses": _v_combination_two_responses,
    "combination:repeat_prompt": _v_combination_repeat_prompt,
    "startend:end_checker": _v_startend_end_checker,
    "startend:quotation": _v_startend_quotation,
    "change_case:english_capital": _v_change_case_english_capital,
    "change_case:english_lowercase": _v_change_case_english_lowercase,
    "change_case:capital_word_frequency": _v_change_case_capital_word_frequency,
    "punctuation:no_comma": _v_punctuation_no_comma,
}


def verify_response(
    instruction_id_list: List[str],
    kwargs_list: List[Dict[str, Any]],
    response: str,
) -> Tuple[bool, List[bool]]:
    """Verify all instructions for a single prompt (strict mode).

    Returns (all_pass, per_instruction_results).
    """
    results = []
    for inst_id, kw in zip(instruction_id_list, kwargs_list):
        fn = _VERIFIERS.get(inst_id)
        if fn is None:
            results.append(True)
            continue
        clean_kw = {k: v for k, v in (kw or {}).items() if v is not None}
        try:
            results.append(fn(response, **clean_kw))
        except Exception:
            results.append(False)
    return all(results), results


# ===================================================================
# IFEval Benchmark Task
# ===================================================================

class IFEvalTask(BenchmarkTask):
    """IFEval instruction-following evaluation task.

    Dataset format: JSON list where each element has:
      - ``"key"`` (int): unique prompt ID
      - ``"prompt"`` (str): the instruction prompt
      - ``"instruction_id_list"`` (list[str]): instruction type IDs
      - ``"kwargs"`` (list[dict]): per-instruction parameters

    The ``reference_answer`` field stores a JSON-serialized dict with
    ``instruction_id_list`` and ``kwargs`` so that ``check_correct``
    can verify the response without needing the original dataset row.
    """

    # -- dataset --------------------------------------------------------

    def load_dataset(
        self, path: str, sample_size: int = -1, seed: int = 42,
    ) -> List[BenchmarkItem]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        items = []
        for entry in raw:
            # Pack verification info into reference_answer as JSON
            ref = json.dumps({
                "instruction_id_list": entry["instruction_id_list"],
                "kwargs": entry["kwargs"],
            }, ensure_ascii=False)
            items.append(BenchmarkItem(
                question=entry["prompt"],
                reference_answer=ref,
                metadata={
                    "key": entry.get("key", 0),
                    "instruction_id_list": entry["instruction_id_list"],
                    "kwargs": entry["kwargs"],
                },
            ))

        if 0 < sample_size < len(items):
            rng = random.Random(seed)
            items = rng.sample(items, sample_size)

        return items

    # -- prompts --------------------------------------------------------

    def build_messages(self, item: BenchmarkItem) -> List[Dict[str, str]]:
        return [{"role": "user", "content": item.question}]

    # -- extraction & checking -----------------------------------------

    def extract_answer(self, response: str, item: BenchmarkItem) -> str:
        # For IFEval the whole response is the answer
        return response

    def check_correct(self, extracted: str, reference: str, **kwargs) -> bool:
        """Check if the response follows all instructions.

        ``reference`` is a JSON string encoding instruction_id_list and kwargs.
        """
        if not extracted:
            return False
        try:
            ref = json.loads(reference)
        except (json.JSONDecodeError, TypeError):
            return False

        all_pass, _ = verify_response(
            ref["instruction_id_list"],
            ref["kwargs"],
            extracted,
        )
        return all_pass

    # -- properties -----------------------------------------------------

    @property
    def name(self) -> str:
        return "ifeval"

    @property
    def question_field(self) -> str:
        return "prompt"

    @property
    def reference_field(self) -> str:
        return "instructions"
