"""Configuration schema for SLM-MUX."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Recommended defaults per benchmark.  When the user does not explicitly
# override confidence / temperature, the framework will use these.
BENCHMARK_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "math500": {
        "confidence_method": "consistency",
        "confidence_params": {"ignore_empty": True},
        "temperature": 0.3,
    },
    "gsm8k": {
        "confidence_method": "consistency",
        "confidence_params": {"ignore_empty": True},
        "temperature": 0.3,
    },
    "gpqa": {
        "confidence_method": "consistency",
        "confidence_params": {"ignore_empty": True},
        "temperature": 0.3,
    },
    "ifeval": {
        "confidence_method": "embedding",
        "confidence_params": {
            "model_name": "BAAI/bge-large-en-v1.5",
            "tau_cluster": 0.90,
        },
        "temperature": 0.7,
    },
    "humaneval": {
        "confidence_method": "embedding",
        "confidence_params": {
            "model_name": "Salesforce/codet5p-110m-embedding",
            "tau_cluster": 0.90,
        },
        "temperature": 0.8,
    },
}


@dataclass
class ProviderConfig:
    """Configuration for a single provider."""

    name: str
    type: str  # together, openai, gemini, vllm, sglang, lmdeploy, hf_endpoint
    api_key_env: str = ""
    base_url: str = ""
    registry_path: str = ""
    max_concurrent: int = 50
    max_qps: float = 0.0
    max_rpm: float = 0.0
    request_timeout: float = 300.0
    max_retries: int = 5
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    id: str  # e.g. "Qwen/Qwen2.5-7B-Instruct-Turbo"
    provider: str  # references ProviderConfig.name
    validation_accuracy: float = 0.0  # for tie-breaking
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceConfig:
    """Configuration for the confidence evaluation method."""

    method: str = "consistency"
    # consistency, verbalized, logprob, reward_model, embedding, learned_router
    params: Dict[str, Any] = field(default_factory=dict)
    # method-specific parameters, e.g.:
    #   consistency: {ignore_empty: true}
    #   verbalized: {model_name: "gpt-4o"}
    #   embedding: {model_name: "text-embedding-3-small"}
    #   learned_router: {classifier_path: "path/to/model.pkl"}


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark to evaluate."""

    name: str = "math500"  # math500, gpqa, gsm8k, ifeval, humaneval
    dataset_path: str = ""
    sample_size: int = -1  # -1 for all
    seed: int = 42


@dataclass
class MUXRunConfig:
    """Configuration for MUX execution."""

    samples_per_model: int = 5
    temperature: float = 0.3
    max_tokens: int = 4096
    tie_breaker: str = "validation_accuracy"
    # validation_accuracy, model_order, random
    num_workers: int = 10
    sc_after_unanimous: bool = True


@dataclass
class OutputConfig:
    """Configuration for output."""

    dir: str = "./results"
    format: str = "json"  # json, jsonl


@dataclass
class MUXConfig:
    """Top-level configuration for SLM-MUX."""

    providers: List[ProviderConfig] = field(default_factory=list)
    models: List[ModelConfig] = field(default_factory=list)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    mux: MUXRunConfig = field(default_factory=MUXRunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def get_provider_config(self, name: str) -> Optional[ProviderConfig]:
        """Look up a provider config by name."""
        for p in self.providers:
            if p.name == name:
                return p
        return None

    def get_model_provider(self, model_id: str) -> Optional[str]:
        """Return the provider name for a given model ID."""
        for m in self.models:
            if m.id == model_id:
                return m.provider
        return None

    def get_model_accuracies(self) -> Dict[str, float]:
        """Return model_id -> validation_accuracy mapping."""
        return {m.id: m.validation_accuracy for m in self.models if m.validation_accuracy > 0}

    def apply_benchmark_defaults(self) -> None:
        """Fill in recommended confidence / temperature settings for the
        selected benchmark, without overriding anything the user explicitly
        set in their config file.

        Call this after loading the config YAML and before running the
        pipeline so that each benchmark uses its best-known parameters
        out of the box.
        """
        defaults = BENCHMARK_DEFAULTS.get(self.benchmark.name)
        if defaults is None:
            return

        # Only override confidence method if the user left it at the
        # generic default ("consistency") and the benchmark wants something
        # different.  This way, an explicit ``confidence.method: xyz`` in
        # the YAML is always respected.
        if self.confidence.method == "consistency" and not self.confidence.params:
            recommended_method = defaults.get("confidence_method")
            if recommended_method and recommended_method != "consistency":
                self.confidence.method = recommended_method
                self.confidence.params = dict(defaults.get("confidence_params", {}))

        # Same logic for temperature: only override if the user left it at
        # the default 0.3.
        if self.mux.temperature == 0.3:
            recommended_temp = defaults.get("temperature")
            if recommended_temp is not None and recommended_temp != 0.3:
                self.mux.temperature = recommended_temp
