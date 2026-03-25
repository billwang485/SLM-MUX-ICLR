"""Live SLM-MUX orchestration CLI.

Replaces the six orchestrator scripts:
  - slm_mux_orchestrator/{math,gpqa,gsm8k}_benchmark.py
  - slm_mux_vllm/{math,gpqa,gsm8k}_benchmark.py
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict

from slm_mux.cli.collect import _get_benchmark, _get_provider
from slm_mux.config.loader import load_config

logger = logging.getLogger(__name__)


def run_mux(args):
    """Entry point for `slm-mux run`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    config = load_config(args.config)
    benchmark = _get_benchmark(config.benchmark.name)

    items = benchmark.load_dataset(
        config.benchmark.dataset_path,
        sample_size=config.benchmark.sample_size,
        seed=config.benchmark.seed,
    )
    logger.info(f"Loaded {len(items)} items for {benchmark.name}")

    model_ids = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else [m.id for m in config.models]
    )
    samples = args.samples if args.samples > 0 else config.mux.samples_per_model

    # Build provider map
    providers: Dict[str, object] = {}
    for model_id in model_ids:
        provider_name = config.get_model_provider(model_id)
        if provider_name is None:
            raise ValueError(f"No provider for model {model_id}")
        if provider_name not in providers:
            providers[model_id] = _get_provider(config, provider_name)
        else:
            providers[model_id] = providers.get(model_id) or _get_provider(config, provider_name)

    # Deduplicate: multiple models may share a provider instance
    provider_instances: Dict[str, object] = {}
    provider_map: Dict[str, object] = {}
    for model_id in model_ids:
        pname = config.get_model_provider(model_id)
        if pname not in provider_instances:
            provider_instances[pname] = _get_provider(config, pname)
        provider_map[model_id] = provider_instances[pname]

    # Build confidence evaluator
    from slm_mux.confidence.consistency import ConsistencyConfidence
    confidence_evaluator = ConsistencyConfidence(
        ignore_empty=config.confidence.params.get("ignore_empty", True)
    )
    if config.confidence.method != "consistency":
        logger.warning(
            f"Confidence method '{config.confidence.method}' requested but "
            f"only 'consistency' is fully supported for live MUX. Using consistency."
        )

    # Build tie-breaker
    from slm_mux.engine.tie_breaker import (
        ModelOrderTieBreaker,
        RandomTieBreaker,
        ValidationAccuracyTieBreaker,
    )
    tb_name = config.mux.tie_breaker
    if tb_name == "validation_accuracy":
        tie_breaker = ValidationAccuracyTieBreaker()
    elif tb_name == "model_order":
        tie_breaker = ModelOrderTieBreaker()
    elif tb_name == "random":
        tie_breaker = RandomTieBreaker()
    else:
        tie_breaker = ValidationAccuracyTieBreaker()

    # Run MUX engine
    from slm_mux.engine.mux import MUXEngine
    engine = MUXEngine(
        providers=provider_map,
        confidence_evaluator=confidence_evaluator,
        benchmark=benchmark,
        models=model_ids,
        samples_per_model=samples,
        temperature=config.mux.temperature,
        max_tokens=config.mux.max_tokens,
        tie_breaker=tie_breaker,
        model_accuracies=config.get_model_accuracies(),
    )

    report = engine.run(items, num_workers=config.mux.num_workers, progress=True)

    # Save results
    output_dir = config.output.dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"mux_{benchmark.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    output_data = {
        "benchmark": benchmark.name,
        "models": model_ids,
        "samples_per_model": samples,
        "temperature": config.mux.temperature,
        "confidence_method": config.confidence.method,
        "tie_breaker": config.mux.tie_breaker,
        "accuracy": report.accuracy,
        "correct_count": report.correct_count,
        "total_count": report.total_count,
        "token_usage": report.total_token_usage,
        "results": [
            {
                benchmark.question_field: r.question,
                "reference_answer": r.reference_answer,
                "final_answer": r.final_answer,
                "is_correct": r.is_correct,
                "selected_model": r.selected_model,
                "confidence_score": r.confidence_score,
                "all_agree": r.all_agree,
            }
            for r in report.results
        ],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(
        f"\n[SLM-MUX] {benchmark.name} | Models: {', '.join(model_ids)}\n"
        f"  Accuracy: {report.accuracy:.2%} ({report.correct_count}/{report.total_count})\n"
        f"  Saved to: {out_path}"
    )
