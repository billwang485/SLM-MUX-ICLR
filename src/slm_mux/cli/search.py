"""Model combination search CLI.

Replaces:
  - rebuttal_testset_scaling_model_types/offline_mux/offline_confacc_select.py
"""

import json
import logging
import os
from typing import Dict

from slm_mux.cli.offline import _autodetect_models

logger = logging.getLogger(__name__)


def run_search(args):
    """Entry point for `slm-mux search`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from slm_mux.engine.offline import OfflineMUX
    from slm_mux.engine.selector import ModelSelector

    benchmark_name = args.benchmark
    data_dir = args.data_dir
    k_min = args.k_min
    k_max = args.k_max
    lambda_c = args.lambda_c
    robust_level = args.robust_level

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

    logger.info(f"Model search: {benchmark_name} | {len(model_ids)} models | k={k_min}-{k_max} | lambda={lambda_c}")

    # Load model data
    offline = OfflineMUX()
    model_data: Dict[str, dict] = {}
    for model_id, path in model_to_path.items():
        model_data[model_id] = offline.load_model_data(path, benchmark_name=benchmark_name)

    # Search
    selector = ModelSelector(lambda_c=lambda_c, robust_level=robust_level)
    best_by_k = selector.search(model_data, model_ids, k_min=k_min, k_max=k_max)

    # Print results
    for k in sorted(best_by_k.keys()):
        rec = best_by_k[k]
        print(
            f"{benchmark_name.upper()} K={k}: score={rec.score:.4f} "
            f"union_acc={rec.union_acc:.4f} | {', '.join(rec.combo)}"
        )

    # Save
    output_dir = args.output_dir or os.path.join(data_dir, "offline_mux")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir,
        f"search_{benchmark_name}_k{k_min}-{k_max}_l{lambda_c}.json",
    )
    out_json = {
        "benchmark": benchmark_name,
        "k_min": k_min,
        "k_max": k_max,
        "lambda_c": lambda_c,
        "robust_level": robust_level,
        "models_scanned": model_ids,
        "best_by_k": {
            str(k): {
                "combo": rec.combo,
                "score": rec.score,
                "union_acc": rec.union_acc,
                "contradiction_count": rec.contradiction_count,
            }
            for k, rec in best_by_k.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {out_path}")
