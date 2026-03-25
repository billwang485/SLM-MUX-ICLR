"""Offline MUX simulation CLI.

Replaces:
  - rebuttal_testset_scaling_model_types/offline_mux/offline_mux_*.py
"""

import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)


def _autodetect_models(data_dir: str) -> Dict[str, str]:
    """Auto-detect model JSONs in a directory. Returns model_id -> file_path."""
    out: Dict[str, str] = {}
    for name in os.listdir(data_dir):
        if not name.endswith(".json") or name == "run_meta.json":
            continue
        model_id = name[:-5].replace("_", "/")
        out[model_id] = os.path.join(data_dir, name)
    if not out:
        raise FileNotFoundError(f"No model JSONs found in {data_dir}")
    return out


def run_offline(args):
    """Entry point for `slm-mux offline`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from slm_mux.confidence.consistency import ConsistencyConfidence
    from slm_mux.engine.offline import OfflineMUX
    from slm_mux.engine.tie_breaker import ValidationAccuracyTieBreaker

    benchmark_name = args.benchmark
    data_dir = args.data_dir
    samples = args.samples
    trials = args.trials
    seed = args.seed

    # Detect models
    if args.models:
        model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        model_to_path = {}
        for m in model_ids:
            fname = m.replace("/", "_") + ".json"
            fpath = os.path.join(data_dir, fname)
            if not os.path.isfile(fpath):
                raise FileNotFoundError(f"Model file not found: {fpath}")
            model_to_path[m] = fpath
    else:
        model_to_path = _autodetect_models(data_dir)
        model_ids = list(model_to_path.keys())

    logger.info(f"Offline MUX: {benchmark_name} | {len(model_ids)} models | {samples} samples | {trials} trials")

    # Load model data
    offline = OfflineMUX(
        confidence_evaluator=ConsistencyConfidence(),
        tie_breaker=ValidationAccuracyTieBreaker(),
    )

    model_data = {}
    model_accuracies = {}
    for model_id, path in model_to_path.items():
        model_data[model_id] = offline.load_model_data(path, benchmark_name=benchmark_name)
        # Read accuracy from the JSON
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_accuracies[model_id] = float(data.get("accuracy", 0.0) or 0.0)

    offline.model_accuracies = model_accuracies

    result = offline.simulate(
        model_data=model_data,
        models=model_ids,
        samples_per_model=samples,
        trials=trials,
        seed=seed,
    )

    # Output
    output_dir = args.output_dir or os.path.join(data_dir, "offline_mux")
    os.makedirs(output_dir, exist_ok=True)

    out_json = {
        "benchmark": benchmark_name,
        "models": model_ids,
        "samples_per_model": samples,
        "trials": trials,
        "seed": seed,
        "accuracy": result.accuracy,
        "num_correct": result.correct,
        "total": result.total,
    }
    out_path = os.path.join(
        output_dir,
        f"offline_mux_{benchmark_name}_s{samples}_t{trials}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(
        f"{benchmark_name.upper()} acc={result.accuracy:.4f} "
        f"({result.correct}/{result.total}) | "
        f"samples={samples} trials={trials} -> {out_path}"
    )
