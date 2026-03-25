# SLM-MUX: Small Language Model Multiplexer

**Confidence-based multi-model routing for small language models at inference time.**

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://openreview.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

SLM-MUX routes each query to the best small language model (SLM) in a pool by
generating multiple samples from each model, estimating per-model confidence,
and selecting the most confident model's answer. This is a **pure inference-time**
method -- no training, no fine-tuning, no additional data needed.

## Key Results

On MATH-500 with 5 SLMs and k=5 samples per model:

| Method | Accuracy |
|--------|----------|
| Best single SLM (Mistral-Small-24B) | 74.0% |
| Average single SLM | 71.3% |
| **SLM-MUX (Consistency Confidence)** | **79.2%** |
| Oracle (upper bound) | 87.2% |

SLM-MUX also supports open-ended benchmarks (IFEval, HumanEval) using
embedding-based confidence scoring.

## How It Works

```
          ┌──────────┐
  Query ──┤          ├── k samples ──► Confidence ──┐
          │  Model 1 │                   Score       │
          └──────────┘                               │
          ┌──────────┐                               │   ┌──────────┐
  Query ──┤          ├── k samples ──► Confidence ───┼──►│  Select  │──► Answer
          │  Model 2 │                   Score       │   │  Highest │
          └──────────┘                               │   └──────────┘
          ┌──────────┐                               │
  Query ──┤          ├── k samples ──► Confidence ──┘
          │  Model N │                   Score
          └──────────┘
```

**Algorithm:**
1. For each query, generate *k* samples independently from each model
2. Evaluate per-model confidence using a chosen method
3. Select the model with the highest confidence
4. Return that model's selected answer

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[all]"
```

## Quick Start

### 1. Configure your models

```yaml
# configs/together.yaml
providers:
  - name: together
    type: together
    api_key_env: TOGETHER_API_KEY

models:
  - id: "Qwen/Qwen2.5-7B-Instruct-Turbo"
    provider: together
  - id: "mistralai/Mistral-Small-24B-Instruct-2501"
    provider: together
  - id: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    provider: together

benchmark:
  name: math500
  dataset_path: data/math_500.json

mux:
  samples_per_model: 5
  temperature: 0.3
```

### 2. Collect samples

```bash
# Collect k=5 samples per model per question
slm-mux collect --config configs/together.yaml
```

### 3. Run offline MUX evaluation

```bash
# Evaluate with majority-vote confidence routing
slm-mux offline \
    --benchmark math500 \
    --data-dir results/ \
    --models Qwen/Qwen2.5-7B-Instruct-Turbo mistralai/Mistral-Small-24B-Instruct-2501 \
    --samples 5 \
    --trials 10
```

### 4. Search for the best model combination

```bash
slm-mux search \
    --benchmark math500 \
    --data-dir results/ \
    --k-min 2 --k-max 5
```

## Supported Benchmarks

| Benchmark | Type | Scoring | Confidence Method |
|-----------|------|---------|-------------------|
| **MATH-500** | Math reasoning | Exact match | Consistency (default) |
| **GSM8K** | Grade school math | Exact match | Consistency (default) |
| **GPQA** | Graduate-level MCQA | Multiple choice | Consistency (default) |
| **IFEval** | Instruction following | Deterministic constraint verification | Embedding (auto-selected) |

Benchmark-specific defaults (confidence method, temperature) are automatically
applied -- just specify the benchmark name and the best-known parameters are
used out of the box.

## Confidence Methods

| Method | Description | Best For |
|--------|-------------|----------|
| **Consistency** | Majority vote over *k* extracted answers | Exact-answer tasks (math, MCQA) |
| **Embedding Similarity** | Cluster responses in embedding space | Open-ended generation (IFEval, code) |
| **Hidden State Probe** | Train MLP on intermediate activations (SAPLMA) | When model internals are accessible |
| **Verbalized** | Ask the model to self-report confidence | Quick estimation |
| **Log-Probability** | Token-level log-probs | When logprobs are available |
| **Reward Model** | External reward model scoring | When a reward model is available |
| **Learned Router** | Pre-trained classifier | When routing training data exists |

## Supported Providers

- **Together AI** -- serverless API
- **vLLM** -- local OpenAI-compatible servers
- **LMDeploy** -- local TurboMind servers
- **SGLang** -- local SGLang servers
- **OpenAI** -- GPT models (for evaluation/comparison)
- **Google Gemini** -- Gemini API
- **HuggingFace Endpoints** -- HF Inference Endpoints

## Project Structure

```
src/slm_mux/
├── engine/         # Core MUX algorithm (live + offline simulation)
├── confidence/     # 7 confidence evaluation methods
├── benchmarks/     # MATH-500, GSM8K, GPQA, IFEval
├── providers/      # LLM provider backends
├── extractors/     # Answer extraction (LaTeX, numeric, MCQA, code)
├── config/         # YAML config loading + per-benchmark defaults
├── evaluation/     # Metrics and LLM-based judging
├── data/           # Dataset loaders
└── cli/            # Command-line interface
```

## Citation

```bibtex
@inproceedings{wang2026slmmux,
  title={SLM-MUX: Confidence-Based Multi-Model Routing for Small Language Models},
  author={Wang, Chenyu and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

MIT
