"""
Embedding-based confidence for open-ended generation tasks.

Two modes of operation:

1. **HuggingFace local model** (paper's method for HumanEval):
   Uses Salesforce/codet5p-110m-embedding or similar model to embed code
   snippets, computes pairwise cosine similarity matrix, finds the largest
   coherent cluster, and uses cluster_size / total as confidence score.

2. **External embedding provider** (API-based):
   Uses an external embedding API (e.g., OpenAI text-embedding-3-small)
   with the same similarity/cluster logic.

Ported from evaluation/analyze_self_consistency_code_embedding.py and
evaluation/build_central_scored_from_embeddings.py.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from .base import ConfidenceEvaluator, ConfidenceResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python similarity helpers (no numpy/torch dependency for basic usage)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _pairwise_similarities(embeddings: List[List[float]]) -> List[Tuple[int, int, float]]:
    """Compute all i < j pairwise cosine similarities. Returns [(i, j, sim)]."""
    n = len(embeddings)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            pairs.append((i, j, sim))
    return pairs


def _cluster_sizes(n: int, pairs: List[Tuple[int, int, float]], tau: float) -> List[int]:
    """
    For each of n items, count how many neighbors have sim >= tau.
    Cluster size for item i = 1 (itself) + number of neighbors.

    This is the paper's method: the item in the largest cluster is selected,
    and cluster_size / n is the confidence score.
    """
    neighbors = [0] * n
    for i, j, sim in pairs:
        if sim >= tau:
            neighbors[i] += 1
            neighbors[j] += 1
    return [1 + neighbors[k] for k in range(n)]


def _centrality_scores(n: int, pairs: List[Tuple[int, int, float]]) -> List[float]:
    """Sum of pairwise similarities for each item (dense centrality)."""
    scores = [0.0] * n
    for i, j, sim in pairs:
        scores[i] += sim
        scores[j] += sim
    return scores


def classify_by_similarities(
    sims: List[float],
    tau_all_same: float = 0.95,
    tau_two_same: float = 0.85,
) -> str:
    """
    Classify pairwise similarities into ALL_SAME / TWO_SAME / ALL_DIFFERENT.

    Ported from analyze_self_consistency_code_embedding.py.
    """
    if not sims:
        return "ALL_DIFFERENT"
    if min(sims) >= tau_all_same:
        return "ALL_SAME"
    sorted_sims = sorted(sims, reverse=True)
    if len(sorted_sims) >= 2 and sorted_sims[0] >= tau_two_same and sorted_sims[1] < tau_two_same:
        return "TWO_SAME"
    return "ALL_DIFFERENT"


# ---------------------------------------------------------------------------
# HuggingFace embedding helper (optional, requires torch + transformers)
# ---------------------------------------------------------------------------

def embed_with_hf_model(
    texts: List[str],
    model_name: str = "BAAI/bge-large-en-v1.5",
    device: str = "auto",
    max_length: int = 512,
    chunk_overlap: int = 64,
) -> List[List[float]]:
    """
    Embed texts using a HuggingFace model with sliding-window chunking.

    Default model: BAAI/bge-large-en-v1.5 (general-purpose text embedding).
    For code tasks, use ``Salesforce/codet5p-110m-embedding``.

    Returns list of L2-normalized embedding vectors.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(dev)
    model.eval()

    # Respect tokenizer's max length
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 100000:
        effective_max = min(max_length, tok_max)
    else:
        effective_max = max_length

    effective_overlap = min(chunk_overlap, effective_max - 1) if effective_max > 1 else 0
    stride = max(1, effective_max - effective_overlap)

    def _embed_one(text: str) -> List[float]:
        encoded = tokenizer(text, truncation=False, padding=False, return_tensors="pt")
        input_ids = encoded["input_ids"][0]
        seq_len = input_ids.size(0)

        if seq_len == 0:
            return [0.0] * 256  # fallback dim

        chunk_embs = []
        with torch.no_grad():
            for start in range(0, seq_len, stride):
                end = min(start + effective_max, seq_len)
                chunk_ids = input_ids[start:end].unsqueeze(0).to(dev)
                attn_mask = torch.ones_like(chunk_ids, device=dev)
                outputs = model(input_ids=chunk_ids, attention_mask=attn_mask)

                if isinstance(outputs, torch.Tensor):
                    emb = outputs
                else:
                    # mean pooling
                    hidden = outputs.last_hidden_state
                    mask_exp = attn_mask.unsqueeze(-1).expand(hidden.size()).float()
                    emb = (hidden * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)

                chunk_embs.append(emb.squeeze(0))

        # Average all chunks, L2-normalize
        avg_emb = torch.stack(chunk_embs).mean(dim=0)
        avg_emb = torch.nn.functional.normalize(avg_emb, p=2, dim=0)
        return avg_emb.cpu().tolist()

    return [_embed_one(t) for t in texts]


# ---------------------------------------------------------------------------
# Main confidence evaluator
# ---------------------------------------------------------------------------

class EmbeddingSimilarityConfidence(ConfidenceEvaluator):
    """
    Embedding-based confidence for open-ended generation (code, text).

    Paper's method (HumanEval):
    1. Embed all K responses using codet5p-110m-embedding
    2. Compute K×K cosine similarity matrix
    3. For each response, count neighbors with sim >= tau_cluster
    4. Confidence = max_cluster_size / K
    5. Selected answer = response in the largest cluster (closest to centroid as tiebreak)

    Supports two embedding backends:
    - "hf": Local HuggingFace model (default: Salesforce/codet5p-110m-embedding)
    - "api": External provider via an embed() method
    """

    def __init__(
        self,
        backend: str = "hf",
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "auto",
        max_length: int = 512,
        chunk_overlap: int = 64,
        tau_cluster: float = 0.90,
        tau_all_same: float = 0.95,
        tau_two_same: float = 0.85,
        embedding_provider: Any = None,
    ):
        """
        Args:
            backend: "hf" for HuggingFace local model, "api" for external provider.
            model_name: Model identifier for embedding.  Default is
                ``BAAI/bge-large-en-v1.5`` (general-purpose).  For code
                tasks, use ``Salesforce/codet5p-110m-embedding``.
            device: torch device (only for backend="hf").
            max_length: Max token length per chunk (only for backend="hf").
            chunk_overlap: Overlap between chunks (only for backend="hf").
            tau_cluster: Similarity threshold for cluster membership.
            tau_all_same: Threshold for ALL_SAME classification.
            tau_two_same: Threshold for TWO_SAME classification.
            embedding_provider: External provider with embed(texts, model) method.
        """
        self._backend = backend
        self._model_name = model_name
        self._device = device
        self._max_length = max_length
        self._chunk_overlap = chunk_overlap
        self._tau_cluster = tau_cluster
        self._tau_all_same = tau_all_same
        self._tau_two_same = tau_two_same
        self._api_provider = embedding_provider

    @property
    def name(self) -> str:
        return "embedding_similarity"

    @property
    def requires_external_model(self) -> bool:
        return True

    def _get_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings via configured backend."""
        if self._backend == "hf":
            try:
                return embed_with_hf_model(
                    texts,
                    model_name=self._model_name,
                    device=self._device,
                    max_length=self._max_length,
                    chunk_overlap=self._chunk_overlap,
                )
            except Exception as e:
                logger.error("HF embedding failed: %s", e)
                return None
        elif self._backend == "api" and self._api_provider is not None:
            try:
                return self._api_provider.embed(texts=texts, model=self._model_name)
            except Exception as e:
                logger.error("API embedding failed: %s", e)
                return None
        else:
            logger.error("No embedding backend configured (backend=%s)", self._backend)
            return None

    def evaluate(
        self,
        responses: List[str],
        extracted_answers: List[str],
        question: Optional[str] = None,
        logprobs: Optional[List[Dict]] = None,
    ) -> ConfidenceResult:
        """
        Evaluate confidence via embedding similarity.

        Algorithm (paper's method):
        1. Embed all K responses
        2. Compute pairwise cosine similarity matrix
        3. For each response, count neighbors with sim >= tau_cluster
        4. Confidence score = max_cluster_size / K
        5. Selected answer = response in largest cluster (centrality tiebreak)
        """
        if not responses:
            return ConfidenceResult(score=0.0, selected_answer="",
                                    metadata={"method": "embedding_similarity", "error": "no responses"})

        if len(responses) == 1:
            return ConfidenceResult(score=1.0,
                                    selected_answer=extracted_answers[0] if extracted_answers else "",
                                    metadata={"method": "embedding_similarity", "note": "single response"})

        embeddings = self._get_embeddings(responses)
        if embeddings is None:
            return ConfidenceResult(score=0.0,
                                    selected_answer=extracted_answers[0] if extracted_answers else "",
                                    metadata={"method": "embedding_similarity", "error": "embedding failed"})

        n = len(embeddings)
        pairs = _pairwise_similarities(embeddings)
        all_sims = [sim for _, _, sim in pairs]

        # Cluster-based confidence (paper's method)
        clusters = _cluster_sizes(n, pairs, self._tau_cluster)
        max_cluster = max(clusters)
        confidence = max_cluster / n

        # Classification label
        label = classify_by_similarities(all_sims, self._tau_all_same, self._tau_two_same)

        # Select best response: largest cluster, tiebreak by centrality
        centrality = _centrality_scores(n, pairs)
        best_idx = 0
        best_cluster = clusters[0]
        best_centrality = centrality[0]
        for idx in range(1, n):
            if clusters[idx] > best_cluster or (
                clusters[idx] == best_cluster and centrality[idx] > best_centrality
            ):
                best_cluster = clusters[idx]
                best_centrality = centrality[idx]
                best_idx = idx

        selected = extracted_answers[best_idx] if best_idx < len(extracted_answers) else ""

        return ConfidenceResult(
            score=confidence,
            selected_answer=selected,
            metadata={
                "method": "embedding_similarity",
                "backend": self._backend,
                "model_name": self._model_name,
                "tau_cluster": self._tau_cluster,
                "label": label,
                "cluster_sizes": clusters,
                "max_cluster_size": max_cluster,
                "mean_pairwise_similarity": sum(all_sims) / len(all_sims) if all_sims else 0.0,
                "min_pairwise_similarity": min(all_sims) if all_sims else 0.0,
                "centrality_scores": centrality,
                "best_index": best_idx,
                "num_responses": n,
            },
        )
