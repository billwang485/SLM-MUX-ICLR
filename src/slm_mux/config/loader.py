"""YAML/JSON configuration loader for SLM-MUX."""

import json
import os
from typing import Any, Dict

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from slm_mux.config.schema import (
    BenchmarkConfig,
    ConfidenceConfig,
    ModelConfig,
    MUXConfig,
    MUXRunConfig,
    OutputConfig,
    ProviderConfig,
)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_raw(path: str) -> Dict[str, Any]:
    """Load a YAML or JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yaml", ".yml")):
            if yaml is None:
                raise ImportError("pyyaml is required for YAML configs: pip install pyyaml")
            return yaml.safe_load(f) or {}
        else:
            return json.load(f)


def _parse_provider(data: Dict[str, Any]) -> ProviderConfig:
    return ProviderConfig(
        name=data.get("name", ""),
        type=data.get("type", ""),
        api_key_env=data.get("api_key_env", ""),
        base_url=data.get("base_url", ""),
        registry_path=data.get("registry_path", ""),
        max_concurrent=int(data.get("max_concurrent", 50)),
        max_qps=float(data.get("max_qps", 0.0)),
        max_rpm=float(data.get("max_rpm", 0.0)),
        request_timeout=float(data.get("request_timeout", 300.0)),
        max_retries=int(data.get("max_retries", 5)),
        extra={k: v for k, v in data.items() if k not in {
            "name", "type", "api_key_env", "base_url", "registry_path",
            "max_concurrent", "max_qps", "max_rpm", "request_timeout", "max_retries",
        }},
    )


def _parse_model(data: Dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        id=data.get("id", ""),
        provider=data.get("provider", ""),
        validation_accuracy=float(data.get("validation_accuracy", 0.0)),
        extra={k: v for k, v in data.items() if k not in {
            "id", "provider", "validation_accuracy",
        }},
    )


def load_config(path: str, overrides: Dict[str, Any] | None = None) -> MUXConfig:
    """
    Load a MUXConfig from a YAML or JSON file.

    Args:
        path: Path to the config file.
        overrides: Optional dict to merge on top of the loaded config.

    Returns:
        Parsed MUXConfig.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = _load_raw(path)
    if overrides:
        raw = _deep_merge(raw, overrides)

    providers = [_parse_provider(p) for p in raw.get("providers", [])]
    models = [_parse_model(m) for m in raw.get("models", [])]

    conf_data = raw.get("confidence", {})
    confidence = ConfidenceConfig(
        method=conf_data.get("method", "consistency"),
        params=conf_data.get("params", {}),
    )

    bench_data = raw.get("benchmark", {})
    benchmark = BenchmarkConfig(
        name=bench_data.get("name", "math500"),
        dataset_path=bench_data.get("dataset_path", ""),
        sample_size=int(bench_data.get("sample_size", -1)),
        seed=int(bench_data.get("seed", 42)),
    )

    mux_data = raw.get("mux", {})
    mux = MUXRunConfig(
        samples_per_model=int(mux_data.get("samples_per_model", 5)),
        temperature=float(mux_data.get("temperature", 0.3)),
        max_tokens=int(mux_data.get("max_tokens", 4096)),
        tie_breaker=mux_data.get("tie_breaker", "validation_accuracy"),
        num_workers=int(mux_data.get("num_workers", 10)),
        sc_after_unanimous=bool(mux_data.get("sc_after_unanimous", True)),
    )

    out_data = raw.get("output", {})
    output = OutputConfig(
        dir=out_data.get("dir", "./results"),
        format=out_data.get("format", "json"),
    )

    cfg = MUXConfig(
        providers=providers,
        models=models,
        confidence=confidence,
        benchmark=benchmark,
        mux=mux,
        output=output,
    )

    # Apply per-benchmark recommended defaults only when the user did not
    # explicitly configure confidence or temperature in their YAML.
    user_set_confidence = "confidence" in raw and "method" in raw.get("confidence", {})
    user_set_temperature = "mux" in raw and "temperature" in raw.get("mux", {})
    if not user_set_confidence and not user_set_temperature:
        cfg.apply_benchmark_defaults()
    elif not user_set_confidence:
        # Only apply confidence defaults, keep user's temperature
        saved_temp = cfg.mux.temperature
        cfg.apply_benchmark_defaults()
        cfg.mux.temperature = saved_temp
    elif not user_set_temperature:
        # Only apply temperature default, keep user's confidence
        saved_conf = (cfg.confidence.method, dict(cfg.confidence.params))
        cfg.apply_benchmark_defaults()
        cfg.confidence.method, cfg.confidence.params = saved_conf

    return cfg
