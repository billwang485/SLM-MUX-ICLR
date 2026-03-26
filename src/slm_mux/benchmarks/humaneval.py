"""HumanEval benchmark task -- code generation evaluation.

Evaluates functional correctness of generated code by executing it against
unit tests.  Uses embedding-based confidence scoring by default since
correct solutions can differ syntactically.

Dataset: OpenAI HumanEval (164 problems).
Scoring: execution-based pass@k.

Reference: Chen et al. (2021), "Evaluating Large Language Models Trained
on Code", arXiv:2107.03374.
"""

import json
import random
import signal
import contextlib
import io
import traceback
from typing import Any, Dict, List, Optional

from slm_mux.benchmarks.base import BenchmarkItem, BenchmarkTask
from slm_mux.extractors.code import CodeExtractor


# ===================================================================
# Execution-based correctness checking
# ===================================================================

class _TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _TimeoutError("Execution timed out")


def _execute_code(code: str, timeout: float = 5.0) -> tuple:
    """Execute code string and return (passed: bool, error: str|None)."""
    try:
        # Set alarm-based timeout (Unix only)
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(timeout) + 1)
        try:
            exec_globals = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, exec_globals)  # noqa: S102
            return True, None
        except _TimeoutError:
            return False, "Timeout"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    except Exception as e:
        # Fallback if signal doesn't work (e.g., non-main thread)
        try:
            exec_globals = {}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, exec_globals)  # noqa: S102
            return True, None
        except Exception as inner_e:
            return False, f"{type(inner_e).__name__}: {inner_e}"


def check_code_correctness(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout: float = 5.0,
) -> bool:
    """Check if a code completion passes the test suite.

    Constructs: prompt + completion + test + check(entry_point)
    and executes it.
    """
    full_code = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
    passed, _ = _execute_code(full_code, timeout=timeout)
    return passed


# ===================================================================
# HumanEval Benchmark Task
# ===================================================================

class HumanEvalTask(BenchmarkTask):
    """HumanEval code generation benchmark.

    Dataset format: JSONL or JSON list where each element has:
      - ``"task_id"`` (str): e.g. "HumanEval/0"
      - ``"prompt"`` (str): function signature + docstring
      - ``"canonical_solution"`` (str): reference solution
      - ``"test"`` (str): unit test code
      - ``"entry_point"`` (str): function name to test

    The ``reference_answer`` field stores a JSON dict with
    ``canonical_solution``, ``test``, and ``entry_point`` so that
    ``check_correct`` can run execution-based evaluation.
    """

    def __init__(self, language: str = "python") -> None:
        self._extractor = CodeExtractor(language=language)

    # -- dataset --------------------------------------------------------

    def load_dataset(
        self, path: str, sample_size: int = -1, seed: int = 42,
    ) -> List[BenchmarkItem]:
        # Support both JSON list and JSONL formats
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content.startswith("["):
            raw = json.loads(content)
        else:
            raw = [json.loads(line) for line in content.split("\n") if line.strip()]

        items = []
        for entry in raw:
            ref = json.dumps({
                "canonical_solution": entry.get("canonical_solution", ""),
                "test": entry.get("test", ""),
                "entry_point": entry.get("entry_point", ""),
                "prompt": entry.get("prompt", ""),
            }, ensure_ascii=False)
            items.append(BenchmarkItem(
                question=entry.get("prompt", ""),
                reference_answer=ref,
                metadata={
                    "task_id": entry.get("task_id", ""),
                    "entry_point": entry.get("entry_point", ""),
                    "test": entry.get("test", ""),
                },
            ))

        if 0 < sample_size < len(items):
            rng = random.Random(seed)
            items = rng.sample(items, sample_size)

        return items

    # -- prompts --------------------------------------------------------

    def build_messages(self, item: BenchmarkItem) -> List[Dict[str, str]]:
        system = (
            "You are an expert Python programmer. Complete the given function. "
            "Return ONLY the function body inside a ```python code block. "
            "Do not include the function signature or docstring."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": item.question},
        ]

    # -- extraction & checking -----------------------------------------

    def extract_answer(self, response: str, item: BenchmarkItem) -> str:
        """Extract code from the response.

        Tries fenced code block extraction first, then falls back to the
        raw response (some models return bare code).
        """
        code = self._extractor.extract(response)
        if code:
            return code
        # Fallback: treat the whole response as code if it looks like Python
        response = response.strip()
        if response and not response.startswith("```"):
            return response
        return ""

    def check_correct(self, extracted: str, reference: str, **kwargs) -> bool:
        """Check if the generated code passes the test suite.

        ``reference`` is a JSON string encoding prompt, test, and entry_point.
        Falls back to string comparison if execution fails.
        """
        if not extracted:
            return False
        try:
            ref = json.loads(reference)
        except (json.JSONDecodeError, TypeError):
            return False

        prompt = ref.get("prompt", "")
        test = ref.get("test", "")
        entry_point = ref.get("entry_point", "")

        if test and entry_point:
            return check_code_correctness(
                prompt=prompt,
                completion=extracted,
                test=test,
                entry_point=entry_point,
                timeout=kwargs.get("timeout", 5.0),
            )

        # Fallback: compare with canonical solution (imprecise)
        canonical = ref.get("canonical_solution", "").strip()
        return extracted.strip() == canonical

    # -- properties -----------------------------------------------------

    @property
    def name(self) -> str:
        return "humaneval"

    @property
    def question_field(self) -> str:
        return "prompt"

    @property
    def reference_field(self) -> str:
        return "solution"
