"""Unified single-model sample collection.

Replaces the three separate scripts:
  - single_model_inference/collect_gpqa.py
  - single_model_inference/collect_gsm8k.py
  - single_model_inference/collect_math.py
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

from tqdm import tqdm

from slm_mux.benchmarks.base import BenchmarkItem, BenchmarkTask
from slm_mux.config.loader import load_config
from slm_mux.config.schema import MUXConfig

logger = logging.getLogger(__name__)


def _get_benchmark(name: str) -> BenchmarkTask:
    """Instantiate a benchmark by name."""
    if name == "math500":
        from slm_mux.benchmarks.math500 import MATH500Task
        return MATH500Task()
    elif name == "gpqa":
        from slm_mux.benchmarks.gpqa import GPQATask
        return GPQATask()
    elif name == "gsm8k":
        from slm_mux.benchmarks.gsm8k import GSM8KTask
        return GSM8KTask()
    elif name == "ifeval":
        from slm_mux.benchmarks.ifeval import IFEvalTask
        return IFEvalTask()
    elif name == "humaneval":
        from slm_mux.benchmarks.humaneval import HumanEvalTask
        return HumanEvalTask()
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def _get_provider(config: MUXConfig, provider_name: str):
    """Instantiate a provider from config."""
    pc = config.get_provider_config(provider_name)
    if pc is None:
        raise ValueError(f"Provider '{provider_name}' not found in config")

    if pc.type == "together":
        from slm_mux.providers.together import TogetherProvider
        return TogetherProvider(
            max_retries=pc.max_retries,
            timeout=(10, int(pc.request_timeout)),
        )
    elif pc.type == "openai":
        from slm_mux.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(
            base_url=pc.base_url or None,
            max_retries=pc.max_retries,
        )
    elif pc.type == "vllm":
        from slm_mux.providers.vllm import VLLMProvider
        return VLLMProvider(
            registry_path=pc.registry_path or None,
            request_timeout=pc.request_timeout,
        )
    elif pc.type == "gemini":
        from slm_mux.providers.gemini import GeminiProvider
        return GeminiProvider(max_retries=pc.max_retries)
    elif pc.type == "sglang":
        from slm_mux.providers.sglang import SGLangProvider
        return SGLangProvider(
            registry_path=pc.registry_path or None,
            request_timeout=pc.request_timeout,
        )
    elif pc.type == "lmdeploy":
        from slm_mux.providers.lmdeploy import LMDeployProvider
        return LMDeployProvider(
            registry_path=pc.registry_path or None,
            request_timeout=pc.request_timeout,
        )
    elif pc.type == "hf_endpoint":
        from slm_mux.providers.hf_endpoint import HFEndpointProvider
        return HFEndpointProvider(base_url=pc.base_url)
    else:
        raise ValueError(f"Unknown provider type: {pc.type}")


def _collect_for_model(
    model_id: str,
    provider,
    benchmark: BenchmarkTask,
    items: List[BenchmarkItem],
    num_samples: int,
    temperature: float,
    max_tokens: int,
    num_workers: int,
) -> Dict[str, Any]:
    """Collect samples for a single model across all items."""

    results: List[Dict[str, Any]] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def process_one(item: BenchmarkItem) -> Dict[str, Any]:
        messages = benchmark.build_messages(item)
        samples = []
        answer_counts: Dict[str, int] = {}
        item_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for idx in range(max(1, num_samples)):
            result = provider.generate(
                model_name=model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            extracted = benchmark.extract_answer(result.content, item)
            if extracted:
                answer_counts[extracted] = answer_counts.get(extracted, 0) + 1

            for k in item_usage:
                item_usage[k] += result.token_usage.get(k, 0)

            samples.append({
                "round": idx + 1,
                "model_response": result.content,
                "extracted_answer": extracted,
                "token_usage": dict(result.token_usage),
                "is_correct": benchmark.check_correct(extracted, item.reference_answer),
            })

        majority_answer = ""
        if answer_counts:
            majority_answer = max(answer_counts.items(), key=lambda kv: kv[1])[0]
        is_correct = benchmark.check_correct(majority_answer, item.reference_answer) if majority_answer else False

        rec: Dict[str, Any] = {
            "model": model_id,
            benchmark.question_field: item.question,
            benchmark.reference_field: item.reference_answer,
            "samples": samples,
            "aggregation": {
                "method": "majority",
                "majority_answer": majority_answer,
                "is_correct": is_correct,
            },
            "token_usage": item_usage,
        }
        if item.choices is not None:
            rec["choices"] = item.choices
        return rec

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_one, item) for item in items]
        pbar = tqdm(
            total=len(futures),
            desc=f"Collecting {model_id}",
            unit="q",
            dynamic_ncols=True,
            ascii=True,
            file=sys.stdout,
        )
        for f in as_completed(futures):
            results.append(f.result())
            pbar.update(1)
        pbar.close()

    correct_count = sum(1 for r in results if r["aggregation"].get("is_correct"))
    total_count = len(results)
    for r in results:
        for k in total_usage:
            total_usage[k] += r["token_usage"].get(k, 0)

    return {
        "model": model_id,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "temperature": temperature,
        "max_new_tokens": max_tokens,
        "sample_size": total_count,
        "accuracy": correct_count / total_count if total_count > 0 else 0.0,
        "correct_count": correct_count,
        "token_usage": total_usage,
        "num_samples": num_samples,
        "responses": results,
    }


def run_collect(args):
    """Entry point for `slm-mux collect`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(args.config)

    benchmark_name = args.benchmark or config.benchmark.name
    benchmark = _get_benchmark(benchmark_name)

    dataset_path = config.benchmark.dataset_path
    sample_size = config.benchmark.sample_size
    seed = config.benchmark.seed
    items = benchmark.load_dataset(dataset_path, sample_size=sample_size, seed=seed)
    logger.info(f"Loaded {len(items)} items from {dataset_path}")

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else [m.id for m in config.models]
    num_samples = args.samples if args.samples > 0 else config.mux.samples_per_model
    output_dir = args.output_dir or config.output.dir
    os.makedirs(output_dir, exist_ok=True)

    for model_id in model_ids:
        provider_name = config.get_model_provider(model_id)
        if provider_name is None:
            logger.error(f"No provider configured for model {model_id}, skipping")
            continue

        provider = _get_provider(config, provider_name)
        logger.info(f"Collecting {num_samples} samples per question for {model_id} via {provider_name}")

        output_data = _collect_for_model(
            model_id=model_id,
            provider=provider,
            benchmark=benchmark,
            items=items,
            num_samples=num_samples,
            temperature=config.mux.temperature,
            max_tokens=config.mux.max_tokens,
            num_workers=config.mux.num_workers,
        )

        out_file = os.path.join(output_dir, f"{model_id.replace('/', '_')}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(
            f"[*] {model_id}: Accuracy={output_data['accuracy']:.2%} "
            f"({output_data['correct_count']}/{output_data['sample_size']}), "
            f"saved to {out_file}"
        )

    logger.info("Collection complete.")
