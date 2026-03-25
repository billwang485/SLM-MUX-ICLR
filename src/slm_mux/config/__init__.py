"""SLM-MUX configuration loading and schema."""

from slm_mux.config.schema import (
    MUXConfig,
    ProviderConfig,
    ModelConfig,
    ConfidenceConfig,
    BenchmarkConfig,
    OutputConfig,
)
from slm_mux.config.loader import load_config

__all__ = [
    "MUXConfig",
    "ProviderConfig",
    "ModelConfig",
    "ConfidenceConfig",
    "BenchmarkConfig",
    "OutputConfig",
    "load_config",
]
